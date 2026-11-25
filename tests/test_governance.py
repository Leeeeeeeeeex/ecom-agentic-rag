"""数据治理流水线离线单元测试。"""

from pathlib import Path

from ecom_rag.governance import govern_reviews, is_logistics_only_review
from ecom_rag.ingest import load_products, load_reviews
from ecom_rag.schemas import Product, RawReview


def test_real_sample_dataset_governance() -> None:
    """验证样本集中低质短评、纯物流水评与重复评论均被准确过滤。"""
    products = load_products(Path("data/raw/products.json"))
    reviews = load_reviews(Path("data/raw/reviews.jsonl"))

    chunks, report = govern_reviews(products, reviews)

    assert report.total_products == 8
    assert report.accessory_products == 2
    assert report.dropped_short == 2  # "很好", "ok good"
    assert report.dropped_logistics_only == 2  # R1007, R6005
    assert report.dropped_duplicate == 2  # R1008, R7004（含首尾空格）
    assert report.dropped_orphan_sku == 0
    assert report.output_chunks == len(chunks)
    assert "PET-CARRIER-AIR" not in report.per_sku_chunks  # 无评论商品用于测试拒答边界


def test_orphan_sku_and_per_sku_cap() -> None:
    """验证孤儿 SKU 被丢弃且单商品超限时优先保留高质量评论。"""
    products = [
        Product(sku_id="SKU-A", title="测试饮水机", category="宠物饮水喂食"),
    ]
    reviews = [
        RawReview(review_id="1", sku_id="SKU-MISSING", content="这是一条归属未知商品的长评论内容测试"),
        RawReview(
            review_id="2",
            sku_id="SKU-A",
            verified_purchase=False,
            helpful_votes=0,
            content="声音还可以，普通体验，长度刚刚够用来做测试样本。",
        ),
        RawReview(
            review_id="3",
            sku_id="SKU-A",
            verified_purchase=True,
            helpful_votes=40,
            content="放卧室实测非常静音，无线水泵拆洗方便，用了两个月不锈钢盘没有黑下巴问题。",
        ),
    ]

    chunks, report = govern_reviews(products, reviews, max_reviews_per_sku=1)

    assert report.dropped_orphan_sku == 1
    assert report.dropped_over_sku_cap == 1
    assert len(chunks) == 1
    assert chunks[0].review_id == "3"  # 高质量已验证购买评论被优先保留


def test_logistics_filter_preserves_real_experience() -> None:
    """同时夸物流和谈使用体验的评论不应被误杀。"""
    assert is_logistics_only_review("物流很快，包装完整，还没拆开用") is True
    assert is_logistics_only_review("物流很快，实测水泵放卧室非常静音不吵人") is False
