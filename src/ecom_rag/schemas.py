"""核心数据模型定义（商品、原始评论、清洗切片与入库治理报告）。"""

from datetime import date
from typing import Any
from pydantic import BaseModel, Field


class Product(BaseModel):
    """商品主数据模型。"""

    sku_id: str
    asin: str | None = None
    title: str
    category: str
    sub_category: str | None = None
    brand: str | None = None
    is_accessory: bool = False
    price_usd: float | None = None
    rating_avg: float | None = None
    review_count: int = 0
    target_pets: list[str] = Field(default_factory=list)
    specs_json: dict[str, Any] = Field(default_factory=dict)


class RawReview(BaseModel):
    """原始评论记录。"""

    review_id: str
    sku_id: str
    rating: int | None = None
    verified_purchase: bool = False
    helpful_votes: int = 0
    review_date: date | None = None
    source_type: str = "review"
    content: str


class CleanedChunk(BaseModel):
    """经过数据治理与切片后待入库的评论片段。"""

    chunk_id: str
    sku_id: str
    review_id: str
    source_type: str = "review"
    content: str
    content_hash: str
    rating: int | None = None
    verified_purchase: bool = False
    helpful_votes: int = 0
    review_date: date | None = None
    quality_score: float


class GovernanceReport(BaseModel):
    """数据治理与入库统计报告（可核验的数据质量证据）。"""

    total_products: int = 0
    accessory_products: int = 0
    total_raw_reviews: int = 0
    dropped_orphan_sku: int = 0
    dropped_short: int = 0
    dropped_logistics_only: int = 0
    dropped_duplicate: int = 0
    dropped_over_sku_cap: int = 0
    output_chunks: int = 0
    deactivated_stale_chunks: int = 0
    per_sku_chunks: dict[str, int] = Field(default_factory=dict)
    elapsed_ms: int = 0
