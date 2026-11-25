"""Redis 限流与问答缓存模块（防刷请求、降低重复调用模型成本）。"""

import hashlib
import json

import redis


class RedisGuardAndCache:
    """基于 Redis 的固定窗口限流与问答结果缓存。"""

    def __init__(self, redis_url: str, rate_limit_per_min: int = 30, cache_ttl: int = 600) -> None:
        self._client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=3)
        self._limit = rate_limit_per_min
        self._ttl = cache_ttl

    def ping(self) -> bool:
        return bool(self._client.ping())

    def check_rate_limit(self, client_key: str) -> tuple[bool, int]:
        """按分钟窗口校验请求频率，返回 (是否放行, 当前窗口计数)。"""
        key = f"ecom_rag:rl:{client_key}"
        pipe = self._client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60, nx=True)
        count, _ = pipe.execute()
        return int(count) <= self._limit, int(count)

    @staticmethod
    def _cache_key(pipeline_version: str, sku_id: str | None, question: str) -> str:
        norm = " ".join(question.strip().lower().split())
        digest = hashlib.sha256(f"{pipeline_version}|{sku_id or ''}|{norm}".encode("utf-8")).hexdigest()[:24]
        return f"ecom_rag:qa:{digest}"

    def get_cached_answer(self, pipeline_version: str, sku_id: str | None, question: str) -> dict | None:
        raw = self._client.get(self._cache_key(pipeline_version, sku_id, question))
        return json.loads(raw) if raw else None

    def set_cached_answer(self, pipeline_version: str, sku_id: str | None, question: str, payload: dict) -> None:
        self._client.setex(
            self._cache_key(pipeline_version, sku_id, question),
            self._ttl,
            json.dumps(payload, ensure_ascii=False),
        )
