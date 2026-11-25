"""可观测与反馈审计模块（PostgreSQL 审计表 + Langfuse v2 HTTP Ingestion 全链路追踪）。"""

import base64
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import psycopg

from ecom_rag.config import Settings


class ObservabilityRecorder:
    """统一记录问答链路耗时、Token 成本、Langfuse Trace/Span/Generation 与线上 Badcase 反馈。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _langfuse_ready(self) -> bool:
        s = self._settings
        return bool(
            s.langfuse_enabled
            and s.langfuse_public_key
            and s.langfuse_secret_key
            and "xxxx" not in s.langfuse_public_key
            and "xxxx" not in s.langfuse_secret_key
        )

    def _send_langfuse_batch(
        self,
        trace_id: str,
        pipeline_version: str,
        sku_id: str | None,
        question: str,
        answer: str,
        abstained: bool,
        retrieved_count: int,
        embed_ms: int,
        retrieve_ms: int,
        generate_ms: int,
        total_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        citation_valid: bool,
        cache_hit: bool,
    ) -> bool:
        """通过 Langfuse v2 标准 /api/public/ingestion 接口上报 Trace + 分阶段 Span + Generation + Score。"""
        now = datetime.now(timezone.utc)
        t_start = now - timedelta(milliseconds=max(1, total_ms))
        t_embed_end = t_start + timedelta(milliseconds=max(0, embed_ms))
        t_ret_end = t_embed_end + timedelta(milliseconds=max(0, retrieve_ms))
        t_gen_end = t_ret_end + timedelta(milliseconds=max(0, generate_ms))

        def iso(dt: datetime) -> str:
            return dt.isoformat().replace("+00:00", "Z")

        batch = [
            {
                "id": uuid.uuid4().hex,
                "type": "trace-create",
                "timestamp": iso(now),
                "body": {
                    "id": trace_id,
                    "timestamp": iso(t_start),
                    "name": f"ecom-rag-{pipeline_version}",
                    "input": {"sku_id": sku_id, "question": question},
                    "output": {"answer": answer, "abstained": abstained},
                    "tags": [
                        pipeline_version,
                        sku_id or "no_sku",
                        "abstained" if abstained else "answered",
                        "cache_hit" if cache_hit else "live",
                    ],
                    "metadata": {
                        "pipeline_version": pipeline_version,
                        "retrieved_count": retrieved_count,
                        "embed_ms": embed_ms,
                        "retrieve_ms": retrieve_ms,
                        "generate_ms": generate_ms,
                        "total_ms": total_ms,
                        "citation_valid": citation_valid,
                        "cache_hit": cache_hit,
                    },
                },
            },
            {
                "id": uuid.uuid4().hex,
                "type": "span-create",
                "timestamp": iso(now),
                "body": {
                    "id": uuid.uuid4().hex,
                    "traceId": trace_id,
                    "name": "embedding_and_retrieval",
                    "startTime": iso(t_start),
                    "endTime": iso(t_ret_end),
                    "input": {"sku_id": sku_id, "question": question},
                    "output": {"retrieved_count": retrieved_count},
                    "metadata": {"embed_ms": embed_ms, "retrieve_ms": retrieve_ms},
                },
            },
            {
                "id": uuid.uuid4().hex,
                "type": "generation-create",
                "timestamp": iso(now),
                "body": {
                    "id": uuid.uuid4().hex,
                    "traceId": trace_id,
                    "name": "llm_generation" if not abstained else "abstain_short_circuit",
                    "model": self._settings.llm_model,
                    "startTime": iso(t_ret_end),
                    "endTime": iso(t_gen_end),
                    "input": question,
                    "output": answer,
                    "usage": {
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                        "unit": "TOKENS",
                    },
                },
            },
            {
                "id": uuid.uuid4().hex,
                "type": "score-create",
                "timestamp": iso(now),
                "body": {
                    "id": uuid.uuid4().hex,
                    "traceId": trace_id,
                    "name": "citation_valid",
                    "value": 1.0 if citation_valid else 0.0,
                },
            },
        ]

        token = base64.b64encode(
            f"{self._settings.langfuse_public_key}:{self._settings.langfuse_secret_key}".encode("utf-8")
        ).decode("ascii")
        url = f"{self._settings.langfuse_host.rstrip('/')}/api/public/ingestion"
        resp = httpx.post(
            url,
            headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
            json={"batch": batch},
            timeout=5.0,
        )
        return resp.status_code in (200, 201, 207)

    def record_qa(
        self,
        pipeline_version: str,
        sku_id: str | None,
        question: str,
        answer: str,
        abstained: bool,
        retrieved_count: int,
        embed_ms: int,
        retrieve_ms: int,
        generate_ms: int,
        total_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
        citation_valid: bool,
        cache_hit: bool = False,
    ) -> str:
        """写入数据库审计表，并在开启 Langfuse 时上报链路 Span。"""
        trace_id = uuid.uuid4().hex

        with psycopg.connect(self._settings.postgres_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO qa_feedback_logs (
                    trace_id, pipeline_version, sku_id, question, answer,
                    abstained, retrieved_count, latency_ms,
                    prompt_tokens, completion_tokens
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    trace_id,
                    pipeline_version,
                    sku_id,
                    question,
                    answer,
                    abstained,
                    retrieved_count,
                    total_ms,
                    prompt_tokens,
                    completion_tokens,
                ),
            )
            conn.commit()

        if self._langfuse_ready():
            try:
                self._send_langfuse_batch(
                    trace_id=trace_id,
                    pipeline_version=pipeline_version,
                    sku_id=sku_id,
                    question=question,
                    answer=answer,
                    abstained=abstained,
                    retrieved_count=retrieved_count,
                    embed_ms=embed_ms,
                    retrieve_ms=retrieve_ms,
                    generate_ms=generate_ms,
                    total_ms=total_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    citation_valid=citation_valid,
                    cache_hit=cache_hit,
                )
            except Exception:
                # 可观测上报失败不应影响主问答链路可用性
                pass

        return trace_id

    def record_feedback(self, trace_id: str, user_feedback: int, badcase_tag: str | None = None) -> int:
        """回写用户点赞/点踩与 Badcase 分类标签，并同步上报 Langfuse 人工评分。"""
        with psycopg.connect(self._settings.postgres_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE qa_feedback_logs
                SET user_feedback = %s, badcase_tag = %s
                WHERE trace_id = %s;
                """,
                (user_feedback, badcase_tag, trace_id),
            )
            updated = cur.rowcount
            conn.commit()

        if updated > 0 and self._langfuse_ready():
            try:
                now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                token = base64.b64encode(
                    f"{self._settings.langfuse_public_key}:{self._settings.langfuse_secret_key}".encode("utf-8")
                ).decode("ascii")
                url = f"{self._settings.langfuse_host.rstrip('/')}/api/public/ingestion"
                httpx.post(
                    url,
                    headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
                    json={
                        "batch": [
                            {
                                "id": uuid.uuid4().hex,
                                "type": "score-create",
                                "timestamp": now_iso,
                                "body": {
                                    "id": uuid.uuid4().hex,
                                    "traceId": trace_id,
                                    "name": "user_feedback",
                                    "value": 1.0 if user_feedback > 0 else 0.0,
                                    "comment": badcase_tag or "",
                                },
                            }
                        ]
                    },
                    timeout=5.0,
                )
            except Exception:
                pass
        return updated
