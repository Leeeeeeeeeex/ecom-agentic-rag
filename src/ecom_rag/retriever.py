"""双版本检索模块：V1 纯向量基线 vs V2 强元数据过滤 + BM25/pgvector RRF 混合检索与质量重排。"""

import time
from dataclasses import dataclass, field

import jieba
import psycopg
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi

from ecom_rag.schemas import Product


@dataclass
class RetrievedChunk:
    """单条召回的评论证据片段。"""

    chunk_id: str
    sku_id: str
    review_id: str
    content: str
    rating: int | None
    verified_purchase: bool
    helpful_votes: int
    quality_score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0


@dataclass
class RetrievalResult:
    """检索阶段统一返回结构（含商品元数据、证据列表与耗时）。"""

    product: Product | None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieve_ms: int = 0
    candidate_pool_size: int = 0


def _tokenize_zh(text: str) -> list[str]:
    """中文 + 英文型号/规格词分词（保留如 2.4GHz、304、IPX7 等关键标记）。"""
    return [tok.strip().lower() for tok in jieba.lcut_for_search(text) if tok.strip()]


def rrf_fuse_and_rerank(
    vector_hits: list[RetrievedChunk],
    bm25_ranked: list[tuple[str, float]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """使用 Reciprocal Rank Fusion (RRF) 融合向量与 BM25 排名，并叠加评论质量加权。"""
    by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in vector_hits}
    rrf_scores: dict[str, float] = {}

    for rank, item in enumerate(vector_hits, start=1):
        rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    bm25_map = dict(bm25_ranked)
    for rank, (chunk_id, bscore) in enumerate(bm25_ranked, start=1):
        if chunk_id in by_id:
            by_id[chunk_id].bm25_score = round(bscore, 4)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    # 归一化 RRF 分并结合语义相似度与评论信息质量分重排
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
    for chunk_id, chunk in by_id.items():
        norm_rrf = rrf_scores.get(chunk_id, 0.0) / max_rrf
        verified_boost = 0.05 if chunk.verified_purchase else 0.0
        chunk.final_score = round(
            0.55 * norm_rrf
            + 0.30 * max(0.0, chunk.vector_score)
            + 0.15 * chunk.quality_score
            + verified_boost,
            4,
        )

    ranked = sorted(by_id.values(), key=lambda c: c.final_score, reverse=True)
    return ranked[:top_k]


class ReviewRetriever:
    """面向 PostgreSQL + pgvector 的 V1/V2 检索器。"""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def get_product(self, sku_id: str | None) -> Product | None:
        """按 sku_id 查询商品主数据。"""
        if not sku_id:
            return None
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT sku_id, asin, title, category, sub_category, brand,
                       is_accessory, price_usd, rating_avg, review_count,
                       target_pets, specs_json
                FROM products
                WHERE sku_id = %s;
                """,
                (sku_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return Product(
            sku_id=row[0],
            asin=row[1],
            title=row[2],
            category=row[3],
            sub_category=row[4],
            brand=row[5],
            is_accessory=row[6],
            price_usd=float(row[7]) if row[7] is not None else None,
            rating_avg=float(row[8]) if row[8] is not None else None,
            review_count=row[9] or 0,
            target_pets=list(row[10] or []),
            specs_json=dict(row[11] or {}),
        )

    def list_products_with_stats(self) -> list[dict]:
        """列出全部商品及已治理入库的评论切片数量。"""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.sku_id, p.title, p.category, p.is_accessory, p.price_usd,
                       COUNT(c.chunk_id) FILTER (WHERE c.is_active) AS active_chunks
                FROM products p
                LEFT JOIN review_chunks c ON p.sku_id = c.sku_id
                GROUP BY p.sku_id
                ORDER BY p.is_accessory ASC, p.sku_id ASC;
                """
            )
            rows = cur.fetchall()
        return [
            {
                "sku_id": r[0],
                "title": r[1],
                "category": r[2],
                "is_accessory": r[3],
                "price_usd": float(r[4]) if r[4] is not None else None,
                "active_chunks": r[5],
            }
            for r in rows
        ]

    def retrieve_v1_pure_vector(
        self,
        query_vec: list[float],
        sku_id: str | None = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        """V1 基线：全库纯向量检索（不按 sku_id 强隔离、不排配件、无 BM25 与重排）。"""
        t0 = time.perf_counter()
        product = self.get_product(sku_id)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, sku_id, review_id, content, rating,
                       verified_purchase, helpful_votes, quality_score,
                       1 - (embedding <=> %s::vector) AS vec_score
                FROM review_chunks
                WHERE is_active = TRUE
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vec, query_vec, top_k),
            )
            rows = cur.fetchall()

        chunks = [
            RetrievedChunk(
                chunk_id=r[0],
                sku_id=r[1],
                review_id=r[2],
                content=r[3],
                rating=r[4],
                verified_purchase=r[5],
                helpful_votes=r[6],
                quality_score=float(r[7]),
                vector_score=round(float(r[8]), 4),
                final_score=round(float(r[8]), 4),
            )
            for r in rows
        ]
        return RetrievalResult(
            product=product,
            chunks=chunks,
            retrieve_ms=int((time.perf_counter() - t0) * 1000),
            candidate_pool_size=len(chunks),
        )

    def retrieve_v2_hybrid_rerank(
        self,
        question: str,
        query_vec: list[float],
        sku_id: str | None = None,
        exclude_accessories: bool = True,
        top_k: int = 5,
        candidate_limit: int = 25,
    ) -> RetrievalResult:
        """V2 生产级检索：强元数据过滤 + pgvector 语义召回 + BM25 词法召回 + RRF 质量重排。"""
        t0 = time.perf_counter()
        product = self.get_product(sku_id)

        with self._connect() as conn, conn.cursor() as cur:
            if sku_id:
                cur.execute(
                    """
                    SELECT c.chunk_id, c.sku_id, c.review_id, c.content, c.rating,
                           c.verified_purchase, c.helpful_votes, c.quality_score,
                           1 - (c.embedding <=> %s::vector) AS vec_score
                    FROM review_chunks c
                    WHERE c.is_active = TRUE AND c.sku_id = %s
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_vec, sku_id, query_vec, candidate_limit),
                )
            else:
                cur.execute(
                    """
                    SELECT c.chunk_id, c.sku_id, c.review_id, c.content, c.rating,
                           c.verified_purchase, c.helpful_votes, c.quality_score,
                           1 - (c.embedding <=> %s::vector) AS vec_score
                    FROM review_chunks c
                    JOIN products p ON p.sku_id = c.sku_id
                    WHERE c.is_active = TRUE
                      AND (%s = FALSE OR p.is_accessory = FALSE)
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_vec, exclude_accessories, query_vec, candidate_limit),
                )
            rows = cur.fetchall()

        if not rows:
            return RetrievalResult(
                product=product,
                chunks=[],
                retrieve_ms=int((time.perf_counter() - t0) * 1000),
                candidate_pool_size=0,
            )

        vector_hits = [
            RetrievedChunk(
                chunk_id=r[0],
                sku_id=r[1],
                review_id=r[2],
                content=r[3],
                rating=r[4],
                verified_purchase=r[5],
                helpful_votes=r[6],
                quality_score=float(r[7]),
                vector_score=round(float(r[8]), 4),
            )
            for r in rows
        ]

        # 在受控候选池上计算中文 BM25，精准召回规格数字与专有术语
        tokenized_docs = [_tokenize_zh(c.content) for c in vector_hits]
        tokenized_query = _tokenize_zh(question)
        bm25 = BM25Okapi(tokenized_docs)
        raw_bm25 = bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(
            [(c.chunk_id, float(score)) for c, score in zip(vector_hits, raw_bm25, strict=True)],
            key=lambda x: x[1],
            reverse=True,
        )

        fused = rrf_fuse_and_rerank(vector_hits, bm25_ranked, top_k=top_k)
        return RetrievalResult(
            product=product,
            chunks=fused,
            retrieve_ms=int((time.perf_counter() - t0) * 1000),
            candidate_pool_size=len(vector_hits),
        )
