"""PostgreSQL + pgvector 持久层（幂等入库、增量更新与逻辑软删除）。"""

import json
from collections import defaultdict

import psycopg
from pgvector.psycopg import register_vector

from ecom_rag.schemas import CleanedChunk, Product


class PostgresRepository:
    """商品主数据与评论向量库仓储。"""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def health_check(self) -> dict:
        """检查数据库连通性与 pgvector 扩展状态。"""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            row = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM products;")
            product_cnt = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM review_chunks WHERE is_active = TRUE;")
            chunk_cnt = cur.fetchone()[0]
        return {
            "pgvector_version": row[0] if row else None,
            "products": product_cnt,
            "active_chunks": chunk_cnt,
        }

    def upsert_products(self, products: list[Product]) -> int:
        """幂等写入/更新商品主数据。"""
        sql = """
            INSERT INTO products (
                sku_id, asin, title, category, sub_category, brand,
                is_accessory, price_usd, rating_avg, review_count,
                specs_json, target_pets, updated_at
            )
            VALUES (
                %(sku_id)s, %(asin)s, %(title)s, %(category)s, %(sub_category)s, %(brand)s,
                %(is_accessory)s, %(price_usd)s, %(rating_avg)s, %(review_count)s,
                %(specs_json)s::jsonb, %(target_pets)s, NOW()
            )
            ON CONFLICT (sku_id) DO UPDATE SET
                asin = EXCLUDED.asin,
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                sub_category = EXCLUDED.sub_category,
                brand = EXCLUDED.brand,
                is_accessory = EXCLUDED.is_accessory,
                price_usd = EXCLUDED.price_usd,
                rating_avg = EXCLUDED.rating_avg,
                review_count = EXCLUDED.review_count,
                specs_json = EXCLUDED.specs_json,
                target_pets = EXCLUDED.target_pets,
                updated_at = NOW();
        """
        with self._connect() as conn, conn.cursor() as cur:
            for p in products:
                params = p.model_dump()
                params["specs_json"] = json.dumps(p.specs_json, ensure_ascii=False)
                cur.execute(sql, params)
            conn.commit()
        return len(products)

    def upsert_chunks_and_sync(
        self,
        chunks: list[CleanedChunk],
        vectors: list[list[float]],
        deactivate_stale: bool = True,
    ) -> tuple[int, int]:
        """增量写入评论切片与向量，并对已下架/已删除的旧切片执行逻辑软删除。"""
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致。")

        upsert_sql = """
            INSERT INTO review_chunks (
                chunk_id, sku_id, review_id, source_type, content, content_hash,
                rating, verified_purchase, helpful_votes, review_date,
                quality_score, is_active, embedding, updated_at
            )
            VALUES (
                %(chunk_id)s, %(sku_id)s, %(review_id)s, %(source_type)s, %(content)s,
                %(content_hash)s, %(rating)s, %(verified_purchase)s, %(helpful_votes)s,
                %(review_date)s, %(quality_score)s, TRUE, %(embedding)s, NOW()
            )
            ON CONFLICT (sku_id, content_hash) DO UPDATE SET
                review_id = EXCLUDED.review_id,
                rating = EXCLUDED.rating,
                verified_purchase = EXCLUDED.verified_purchase,
                helpful_votes = EXCLUDED.helpful_votes,
                review_date = EXCLUDED.review_date,
                quality_score = EXCLUDED.quality_score,
                is_active = TRUE,
                embedding = EXCLUDED.embedding,
                updated_at = NOW();
        """

        hashes_by_sku: dict[str, list[str]] = defaultdict(list)
        deactivated = 0

        with self._connect() as conn, conn.cursor() as cur:
            for chunk, vec in zip(chunks, vectors, strict=True):
                params = chunk.model_dump()
                params["embedding"] = vec
                cur.execute(upsert_sql, params)
                hashes_by_sku[chunk.sku_id].append(chunk.content_hash)

            if deactivate_stale:
                for sku_id, active_hashes in hashes_by_sku.items():
                    cur.execute(
                        """
                        UPDATE review_chunks
                        SET is_active = FALSE, updated_at = NOW()
                        WHERE sku_id = %s
                          AND is_active = TRUE
                          AND NOT (content_hash = ANY(%s));
                        """,
                        (sku_id, active_hashes),
                    )
                    deactivated += cur.rowcount

            conn.commit()
        return len(chunks), deactivated
