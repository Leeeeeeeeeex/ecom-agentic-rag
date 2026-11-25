"""商品与评论数据治理及入库 CLI。

用法示例：
  1) 仅做离线数据治理检查（不连库、不调 Embedding API）：
     python -m ecom_rag.ingest --dry-run
  2) 用真实 Embedding API 入库到虚拟机 PostgreSQL + pgvector：
     python -m ecom_rag.ingest
  3) 先不配 API Key，用确定性测试向量验证虚拟机入库链路：
     python -m ecom_rag.ingest --mock-embedding
"""

import argparse
import json
import time
from pathlib import Path

from ecom_rag.config import get_settings
from ecom_rag.db import PostgresRepository
from ecom_rag.embeddings import (
    DeterministicHashEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from ecom_rag.governance import govern_reviews
from ecom_rag.schemas import Product, RawReview


def load_products(path: Path) -> list[Product]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Product.model_validate(item) for item in data]


def load_reviews(path: Path) -> list[RawReview]:
    reviews: list[RawReview] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        reviews.append(RawReview.model_validate_json(line))
    return reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="跨境电商评论数据治理与 pgvector 入库流水线")
    parser.add_argument("--products", type=Path, default=Path("data/raw/products.json"))
    parser.add_argument("--reviews", type=Path, default=Path("data/raw/reviews.jsonl"))
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-per-sku", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="只执行清洗治理并打印报告，不写库")
    parser.add_argument(
        "--mock-embedding",
        action="store_true",
        help="使用本地确定性伪向量验证入库链路（生产/评测请勿开启）",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    products = load_products(args.products)
    raw_reviews = load_reviews(args.reviews)
    chunks, report = govern_reviews(
        products=products,
        raw_reviews=raw_reviews,
        min_chars=args.min_chars,
        max_reviews_per_sku=args.max_per_sku,
    )

    if args.dry_run:
        report.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print("=== [DRY-RUN] 数据治理报告（未写库）===")
        print(report.model_dump_json(indent=2))
        return

    settings = get_settings()
    if args.mock_embedding:
        embedder = DeterministicHashEmbeddingProvider(dim=settings.embedding_dim)
    else:
        embedder = OpenAICompatibleEmbeddingProvider(settings)

    vectors, embed_ms = embedder.embed_texts([c.content for c in chunks])
    repo = PostgresRepository(settings.postgres_dsn)
    repo.upsert_products(products)
    upserted, deactivated = repo.upsert_chunks_and_sync(chunks, vectors, deactivate_stale=True)

    report.deactivated_stale_chunks = deactivated
    report.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    health = repo.health_check()

    print("=== 入库与治理完成 ===")
    print(f"Embedding 耗时: {embed_ms} ms | 写入切片: {upserted} | 软删除旧切片: {deactivated}")
    print(f"数据库状态: {json.dumps(health, ensure_ascii=False)}")
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
