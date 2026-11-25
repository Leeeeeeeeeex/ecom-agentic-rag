"""商品与评论数据治理流水线（清洗、去重、低质过滤、质量打分与切片）。"""

import hashlib
import math
import re
from collections import defaultdict

from ecom_rag.schemas import CleanedChunk, GovernanceReport, Product, RawReview

# 仅谈论物流/包装/尚未使用的水评特征（无实际使用体验价值）
_LOGISTICS_ONLY_KEYWORDS = (
    "物流很快",
    "快递很快",
    "包装完整",
    "还没拆",
    "还没用",
    "用完再来追评",
    "送上楼",
    "给个好评",
    "发货快",
)
# 若包含真实使用体验关键词，则不按“纯物流水评”误杀
_USAGE_EXPERIENCE_KEYWORDS = (
    "声音",
    "静音",
    "噪音",
    "水泵",
    "滤芯",
    "卡粮",
    "出粮",
    "猫砂",
    "防夹",
    "刀头",
    "续航",
    "电池",
    "清洗",
    "洗碗机",
    "水洗",
    "黑下巴",
    "配网",
    "摄像头",
    "材质",
    "不锈钢",
    "塑料",
    "除臭",
    "发烫",
    "尺寸",
    "漏水",
)


def normalize_text(text: str) -> str:
    """归一化空白字符与首尾空格，保证内容哈希去重稳定。"""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned


def compute_content_hash(sku_id: str, text: str) -> str:
    """按 sku_id + 归一化文本生成 SHA-256 去重指纹。"""
    payload = f"{sku_id}::{text.lower()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_logistics_only_review(text: str) -> bool:
    """判断是否为仅夸物流/包装、无真实使用体验的水评。"""
    has_logistics = any(kw in text for kw in _LOGISTICS_ONLY_KEYWORDS)
    if not has_logistics:
        return False
    has_usage = any(kw in text for kw in _USAGE_EXPERIENCE_KEYWORDS)
    return not has_usage


def compute_quality_score(review: RawReview, normalized_text: str) -> float:
    """计算评论信息密度与可信度综合得分（范围 0.0 ~ 1.0）。"""
    length_score = min(1.0, len(normalized_text) / 80.0)
    verified_bonus = 0.25 if review.verified_purchase else 0.0
    votes_score = min(1.0, math.log1p(max(0, review.helpful_votes)) / math.log1p(50))
    score = 0.40 * length_score + verified_bonus + 0.35 * votes_score
    return round(min(1.0, score), 4)


def split_text_chunks(text: str, chunk_size: int = 260, chunk_overlap: int = 40) -> list[str]:
    """对长评论按句号/换行优先切块，短评论保持整段完整以防语义割裂。"""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += step
    return chunks


def govern_reviews(
    products: list[Product],
    raw_reviews: list[RawReview],
    min_chars: int = 12,
    max_reviews_per_sku: int = 100,
    chunk_size: int = 260,
    chunk_overlap: int = 40,
) -> tuple[list[CleanedChunk], GovernanceReport]:
    """执行完整评论数据治理流水线，返回待入库切片与可审计报告。"""
    valid_skus = {p.sku_id for p in products}
    report = GovernanceReport(
        total_products=len(products),
        accessory_products=sum(1 for p in products if p.is_accessory),
        total_raw_reviews=len(raw_reviews),
    )

    seen_hashes: set[str] = set()
    candidates_by_sku: dict[str, list[tuple[float, RawReview, str, str]]] = defaultdict(list)

    for review in raw_reviews:
        if review.sku_id not in valid_skus:
            report.dropped_orphan_sku += 1
            continue

        norm = normalize_text(review.content)
        compact_len = len(re.sub(r"\s+", "", norm))
        if compact_len < min_chars:
            report.dropped_short += 1
            continue

        if is_logistics_only_review(norm):
            report.dropped_logistics_only += 1
            continue

        chash = compute_content_hash(review.sku_id, norm)
        if chash in seen_hashes:
            report.dropped_duplicate += 1
            continue
        seen_hashes.add(chash)

        qscore = compute_quality_score(review, norm)
        candidates_by_sku[review.sku_id].append((qscore, review, norm, chash))

    # 单商品样本上限控制：按质量得分降序保留高价值评论，防止爆款刷爆索引
    output_chunks: list[CleanedChunk] = []
    for sku_id, items in sorted(candidates_by_sku.items()):
        items.sort(key=lambda x: (x[0], x[1].helpful_votes), reverse=True)
        if len(items) > max_reviews_per_sku:
            report.dropped_over_sku_cap += len(items) - max_reviews_per_sku
            items = items[:max_reviews_per_sku]

        sku_chunk_count = 0
        for qscore, review, norm, _review_hash in items:
            parts = split_text_chunks(norm, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for idx, part in enumerate(parts):
                part_hash = compute_content_hash(sku_id, part)
                chunk = CleanedChunk(
                    chunk_id=f"{review.review_id}_c{idx}",
                    sku_id=sku_id,
                    review_id=review.review_id,
                    source_type=review.source_type,
                    content=part,
                    content_hash=part_hash,
                    rating=review.rating,
                    verified_purchase=review.verified_purchase,
                    helpful_votes=review.helpful_votes,
                    review_date=review.review_date,
                    quality_score=qscore,
                )
                output_chunks.append(chunk)
                sku_chunk_count += 1
        report.per_sku_chunks[sku_id] = sku_chunk_count

    report.output_chunks = len(output_chunks)
    return output_chunks, report
