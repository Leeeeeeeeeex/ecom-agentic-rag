"""FastAPI 生产问答服务入口（含参数校验、Redis 限流/缓存、V1/V2 切换与反馈回流）。"""

import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ecom_rag.cache import RedisGuardAndCache
from ecom_rag.config import get_settings
from ecom_rag.db import PostgresRepository
from ecom_rag.embeddings import OpenAICompatibleEmbeddingProvider
from ecom_rag.graph import AgenticRAGService
from ecom_rag.observability import ObservabilityRecorder
from ecom_rag.retriever import ReviewRetriever

_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="跨境电商生产级 Agentic RAG 问答服务",
    version="0.2.0",
    description="支持 V1 纯向量基线与 V2 混合检索+证据门控对比、Redis 缓存限流、Langfuse 追踪与 Badcase 回流。",
)


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """返回纯中文交互式前端问答工作台。"""
    return FileResponse(_STATIC_DIR / "index.html")


class QARequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=300, description="买家或导购提问")
    sku_id: str | None = Field(default=None, description="目标商品 SKU（推荐传入以防跨型号串品）")
    pipeline_version: Literal["v1_vector", "v2_agentic_hybrid"] = Field(
        default="v2_agentic_hybrid", description="检索与编排版本"
    )
    top_k: int = Field(default=5, ge=1, le=10)
    use_cache: bool = Field(default=True)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=8)
    user_feedback: Literal[-1, 1] = Field(..., description="1=满意, -1=不满意")
    badcase_tag: Literal["wrong_sku", "hallucination", "missed_recall", "over_abstain", "other"] | None = None


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    pg = PostgresRepository(settings.postgres_dsn).health_check()
    redis_ok = RedisGuardAndCache(settings.redis_url).ping()
    return {
        "status": "ok",
        "postgres": pg,
        "redis": redis_ok,
        "langfuse_url": settings.langfuse_host,
    }


@app.get("/products")
def list_products() -> dict:
    settings = get_settings()
    retriever = ReviewRetriever(settings.postgres_dsn)
    return {"items": retriever.list_products_with_stats()}


@app.post("/api/v1/qa")
def ask_question(req: QARequest, request: Request) -> dict:
    t0 = time.perf_counter()
    settings = get_settings()
    cache = RedisGuardAndCache(
        settings.redis_url,
        rate_limit_per_min=settings.rate_limit_per_min,
        cache_ttl=settings.cache_ttl_seconds,
    )

    client_ip = request.client.host if request.client else "unknown"
    allowed, current_cnt = cache.check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁（当前分钟已请求 {current_cnt} 次，上限 {settings.rate_limit_per_min} 次）。",
        )

    if req.use_cache:
        cached = cache.get_cached_answer(req.pipeline_version, req.sku_id, req.question)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

    try:
        embedder = OpenAICompatibleEmbeddingProvider(settings)
        retriever = ReviewRetriever(settings.postgres_dsn)
        rag = AgenticRAGService(settings=settings, retriever=retriever, embedder=embedder)
        state = rag.ask(
            question=req.question,
            sku_id=req.sku_id,
            pipeline_version=req.pipeline_version,
            top_k=req.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    total_ms = int((time.perf_counter() - t0) * 1000)
    chunks = state.get("chunks", [])
    recorder = ObservabilityRecorder(settings)
    trace_id = recorder.record_qa(
        pipeline_version=req.pipeline_version,
        sku_id=req.sku_id,
        question=req.question,
        answer=state.get("answer", ""),
        abstained=bool(state.get("abstained", False)),
        retrieved_count=len(chunks),
        embed_ms=int(state.get("embed_ms", 0)),
        retrieve_ms=int(state.get("retrieve_ms", 0)),
        generate_ms=int(state.get("generate_ms", 0)),
        total_ms=total_ms,
        prompt_tokens=int(state.get("prompt_tokens", 0)),
        completion_tokens=int(state.get("completion_tokens", 0)),
        citation_valid=bool(state.get("citation_valid", True)),
        cache_hit=False,
    )

    response_payload = {
        "trace_id": trace_id,
        "pipeline_version": req.pipeline_version,
        "cache_hit": False,
        "abstained": bool(state.get("abstained", False)),
        "answer": state.get("answer", ""),
        "citation_valid": bool(state.get("citation_valid", True)),
        "cited_review_ids": state.get("cited_review_ids", []),
        "invalid_citations": state.get("invalid_citations", []),
        "product": state.get("product_dict"),
        "evidence_chunks": [
            {
                "review_id": c.review_id,
                "sku_id": c.sku_id,
                "rating": c.rating,
                "verified_purchase": c.verified_purchase,
                "helpful_votes": c.helpful_votes,
                "vector_score": c.vector_score,
                "bm25_score": c.bm25_score,
                "final_score": c.final_score,
                "content": c.content,
            }
            for c in chunks
        ],
        "metrics": {
            "embed_ms": int(state.get("embed_ms", 0)),
            "retrieve_ms": int(state.get("retrieve_ms", 0)),
            "generate_ms": int(state.get("generate_ms", 0)),
            "total_ms": total_ms,
            "prompt_tokens": int(state.get("prompt_tokens", 0)),
            "completion_tokens": int(state.get("completion_tokens", 0)),
        },
    }

    if req.use_cache:
        cache.set_cached_answer(req.pipeline_version, req.sku_id, req.question, response_payload)
    return response_payload


@app.post("/api/v1/feedback")
def submit_feedback(req: FeedbackRequest) -> dict:
    settings = get_settings()
    recorder = ObservabilityRecorder(settings)
    updated = recorder.record_feedback(req.trace_id, req.user_feedback, req.badcase_tag)
    if updated == 0:
        raise HTTPException(status_code=404, detail=f"未找到 trace_id={req.trace_id} 的问答记录。")
    return {"status": "recorded", "trace_id": req.trace_id, "badcase_tag": req.badcase_tag}
