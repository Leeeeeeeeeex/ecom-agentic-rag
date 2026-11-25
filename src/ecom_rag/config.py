"""全局配置管理模块（基于 pydantic-settings 从 .env 及系统环境变量加载）。"""

import os
from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用运行与外部依赖配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 虚拟机基础设施
    vm_host: str = "192.168.150.130"
    postgres_dsn: str = "postgresql://rag_user:rag_password_change_me@192.168.150.130:5432/ecom_rag"
    redis_url: str = "redis://192.168.150.130:6379/0"

    # LLM 配置（支持自动回退读取系统环境变量 DEEPSEEK_API_KEY）
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # Embedding 配置（支持自动回退读取系统环境变量 DASHSCOPE_API_KEY）
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024

    # Langfuse 可观测配置
    langfuse_enabled: bool = False
    langfuse_host: str = "http://192.168.150.130:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # 服务限流与缓存
    app_env: str = "dev"
    rate_limit_per_min: int = 30
    cache_ttl_seconds: int = 600

    @model_validator(mode="after")
    def _resolve_provider_keys(self) -> "Settings":
        """若 .env 未显式填写 LLM_API_KEY / EMBEDDING_API_KEY，自动读取标准环境变量。"""
        if not self.llm_api_key or "please-replace" in self.llm_api_key:
            self.llm_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not self.embedding_api_key or "please-replace" in self.embedding_api_key:
            self.embedding_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取单例配置对象。"""
    return Settings()
