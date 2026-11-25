"""Embedding 向量化客户端（支持 OpenAI 兼容 API 与离线确定性测试 Stub）。"""

import hashlib
import math
import time
from typing import Protocol

from openai import OpenAI

from ecom_rag.config import Settings


class EmbeddingProvider(Protocol):
    """向量生成接口协议。"""

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """批量将文本转为向量，返回 (vectors, elapsed_ms)。"""
        ...


class OpenAICompatibleEmbeddingProvider:
    """基于 OpenAI 兼容协议的 Embedding 客户端（兼容通义千问 / 硅基流动 / OpenAI）。"""

    def __init__(self, settings: Settings, batch_size: int = 10) -> None:
        if not settings.embedding_api_key or "please-replace" in settings.embedding_api_key:
            raise ValueError("请先在 .env 中填写有效的 EMBEDDING_API_KEY。")
        self._client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """分批请求远程 Embedding API 并记录总耗时。"""
        if not texts:
            return [], 0
        t0 = time.perf_counter()
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            kwargs: dict = {"model": self._model, "input": batch}
            # 部分模型（如 text-embedding-v3 / text-embedding-3-*）支持显式指定维度
            if self._dim > 0 and ("v3" in self._model or "embedding-3" in self._model):
                kwargs["dimensions"] = self._dim
            resp = self._client.embeddings.create(**kwargs)
            ordered = sorted(resp.data, key=lambda item: item.index)
            for item in ordered:
                if len(item.embedding) != self._dim:
                    raise ValueError(
                        f"Embedding 维度不匹配：模型返回 {len(item.embedding)} 维，"
                        f"但 EMBEDDING_DIM={self._dim}（需与数据库 vector({self._dim}) 一致）。"
                    )
                all_vectors.append(item.embedding)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return all_vectors, elapsed_ms


class DeterministicHashEmbeddingProvider:
    """仅用于离线单测与本地无 Key 联调测试的确定性伪向量生成器。"""

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    def embed_texts(self, texts: list[str]) -> tuple[list[list[float]], int]:
        t0 = time.perf_counter()
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for idx in range(len(text)):
                gram = text[idx : idx + 2]
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self._dim
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                vec[bucket] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors, int((time.perf_counter() - t0) * 1000)
