"""Application configuration.

One frozen ``Settings`` object, validated at import time of the container, is the only
source of configuration in the process. ``os.getenv`` is banned repo-wide by a ruff rule
so that no module can grow a hidden second source of truth.

Validation is fail-fast and deliberately strict: a service that boots with a broken
configuration and dies on the first user request is strictly worse than one that never
boots, because the first failure is then a customer's problem instead of a deploy's.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production", "test"]
VectorBackend = Literal["qdrant", "pgvector", "memory"]
StorageBackend = Literal["s3", "local"]
EmbeddingProviderName = Literal["bge_tei", "openai", "voyage", "cohere", "fake"]
RerankerProviderName = Literal["bge_tei", "cross_encoder", "cohere", "noop"]
LLMProviderName = Literal["openai", "anthropic", "gemini"]
SecretsProviderName = Literal["env", "aws", "vault"]

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Every knob the process has, grouped as in ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=(_PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    # ── Core ────────────────────────────────────────────────────────────────
    app_env: Environment = "development"
    app_name: str = "Aegis Assistant"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"
    secret_key: SecretStr = SecretStr("")
    cors_origins: str = "http://localhost:3000"
    public_base_url: str = "http://localhost:8000"

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://aegis:aegis@localhost:5432/aegis"
    database_pool_size: Annotated[int, Field(ge=1, le=100)] = 20
    database_max_overflow: Annotated[int, Field(ge=0, le=100)] = 10
    database_pool_timeout: Annotated[int, Field(ge=1, le=60)] = 10
    database_echo: bool = False

    # ── Redis ───────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_queue_db: int = 1
    redis_max_connections: Annotated[int, Field(ge=1, le=1000)] = 50
    cache_namespace: str = "aegis"

    # ── Vector store ────────────────────────────────────────────────────────
    vector_backend: VectorBackend = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_prefer_grpc: bool = False
    qdrant_quantization: bool = True
    qdrant_on_disk_payload: bool = True

    # ── Object storage ──────────────────────────────────────────────────────
    storage_backend: StorageBackend = "s3"
    local_storage_path: Path = Path("./var/objects")
    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_region: str = "us-east-1"
    s3_bucket: str = "aegis-documents"
    s3_access_key_id: SecretStr = SecretStr("")
    s3_secret_access_key: SecretStr = SecretStr("")
    s3_presign_ttl_seconds: Annotated[int, Field(ge=60, le=3600)] = 900
    s3_force_path_style: bool = True

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProviderName = "bge_tei"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: Annotated[int, Field(ge=64, le=8192)] = 1024
    embedding_batch_size: Annotated[int, Field(ge=1, le=512)] = 64
    embedding_max_concurrency: Annotated[int, Field(ge=1, le=32)] = 4
    embedding_cache_ttl_seconds: int = 604_800
    tei_embed_url: str = "http://localhost:8080"

    reranker_provider: RerankerProviderName = "bge_tei"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    tei_rerank_url: str = "http://localhost:8081"

    # ── LLM ─────────────────────────────────────────────────────────────────
    llm_provider: LLMProviderName = "openai"
    llm_model: str = "gpt-5.1"
    llm_fallback_chain: str = ""
    llm_temperature: Annotated[float, Field(ge=0.0, le=0.3)] = 0.1
    llm_max_output_tokens: Annotated[int, Field(ge=64, le=32_000)] = 2000
    llm_timeout_seconds: Annotated[int, Field(ge=5, le=600)] = 90
    intent_model: str = "gpt-5-mini"

    # Overridable so that any OpenAI-compatible endpoint — vLLM, Ollama, TGI, Groq,
    # Azure — is reachable without a second adapter.
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    voyage_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None

    # ── Retrieval ───────────────────────────────────────────────────────────
    retrieval_top_k_dense: Annotated[int, Field(ge=1, le=500)] = 40
    retrieval_top_k_sparse: Annotated[int, Field(ge=0, le=500)] = 40
    retrieval_rrf_k: Annotated[int, Field(ge=1, le=1000)] = 60
    rerank_top_n: Annotated[int, Field(ge=1, le=50)] = 8
    confidence_min_top_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35
    confidence_min_supporting: Annotated[int, Field(ge=1, le=20)] = 2
    confidence_min_mean_top3: Annotated[float, Field(ge=0.0, le=1.0)] = 0.28
    confidence_min_entity_coverage: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    context_token_budget: Annotated[int, Field(ge=500, le=200_000)] = 10_000
    prompt_token_cap: Annotated[int, Field(ge=1000, le=400_000)] = 16_000
    history_token_budget: Annotated[int, Field(ge=0, le=50_000)] = 2000
    summary_token_budget: Annotated[int, Field(ge=0, le=10_000)] = 600

    # ── Ingestion ───────────────────────────────────────────────────────────
    max_upload_bytes: Annotated[int, Field(ge=1024, le=2 * 1024**3)] = 209_715_200
    chunk_size: Annotated[int, Field(ge=100, le=4000)] = 800
    chunk_overlap_pct: Annotated[int, Field(ge=0, le=50)] = 15
    chunk_min_tokens: Annotated[int, Field(ge=50, le=2000)] = 250
    chunk_max_tokens: Annotated[int, Field(ge=100, le=8000)] = 1400
    chunk_strategy: Literal["adaptive", "recursive", "semantic", "markdown"] = "adaptive"
    contextual_headers: bool = True
    ocr_enabled: bool = True
    ocr_dpi: Annotated[int, Field(ge=72, le=600)] = 300
    ocr_languages: str = "eng"
    ocr_min_chars_per_page: Annotated[int, Field(ge=0, le=5000)] = 100
    malware_scan_enabled: bool = False
    clamav_host: str = "clamav"
    clamav_port: int = 3310
    ingest_max_attempts: Annotated[int, Field(ge=1, le=10)] = 3
    default_language: str = "en"

    # ── Authentication ──────────────────────────────────────────────────────
    access_token_ttl_minutes: Annotated[int, Field(ge=1, le=1440)] = 15
    refresh_token_ttl_days: Annotated[int, Field(ge=1, le=90)] = 14
    jwt_algorithm: Literal["HS256", "RS256"] = "HS256"
    jwt_private_key_path: Path | None = None
    jwt_public_key_path: Path | None = None
    password_min_length: Annotated[int, Field(ge=8, le=128)] = 12
    max_failed_logins: Annotated[int, Field(ge=1, le=20)] = 5
    lockout_minutes: Annotated[int, Field(ge=1, le=1440)] = 15
    mfa_enabled: bool = True
    mfa_required_roles: str = "admin,manager"
    mfa_issuer: str = "Aegis"
    guest_access_enabled: bool = True
    secrets_provider: SecretsProviderName = "env"

    # ── Rate limits ─────────────────────────────────────────────────────────
    rate_limit_enabled: bool = True
    rate_limit_login_per_15min: int = 10
    rate_limit_chat_per_min_guest: int = 5
    rate_limit_chat_per_min_customer: int = 20
    rate_limit_chat_per_min_internal: int = 60
    rate_limit_chat_per_min_admin: int = 120
    rate_limit_upload_per_hour: int = 60
    rate_limit_ticket_per_hour: int = 3
    max_concurrent_streams_per_user: int = 3

    # ── Observability ───────────────────────────────────────────────────────
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "aegis-api"
    trace_sample_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    metrics_enabled: bool = True
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None

    # ── Derived ─────────────────────────────────────────────────────────────

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mfa_required_role_list(self) -> list[str]:
        return [r.strip().lower() for r in self.mfa_required_roles.split(",") if r.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chunk_overlap_tokens(self) -> int:
        return max(0, round(self.chunk_size * self.chunk_overlap_pct / 100))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def redis_queue_url(self) -> str:
        """arq needs its own logical database so a `FLUSHDB` on the cache cannot eat jobs."""
        base, _, _ = self.redis_url.rpartition("/")
        return f"{base or self.redis_url}/{self.redis_queue_db}"

    def fallback_models(self) -> list[tuple[str, str]]:
        """``"anthropic:claude-x,gemini:y"`` → ``[("anthropic", "claude-x"), ("gemini", "y")]``."""
        pairs: list[tuple[str, str]] = []
        for entry in self.llm_fallback_chain.split(","):
            entry = entry.strip()
            if not entry:
                continue
            provider, _, model = entry.partition(":")
            if provider and model:
                pairs.append((provider.strip(), model.strip()))
        return pairs

    # ── Validation ──────────────────────────────────────────────────────────

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver "
                "(postgresql+asyncpg://...); a sync driver blocks the event loop"
            )
        return v

    @field_validator("cors_origins")
    @classmethod
    def _no_wildcard_with_credentials(cls, v: str) -> str:
        if "*" in v:
            raise ValueError(
                "CORS_ORIGINS cannot contain '*': the API is used with credentials, "
                "so a wildcard origin is both invalid per spec and unsafe"
            )
        return v

    @model_validator(mode="after")
    def _validate_cross_field(self) -> Self:
        errors: list[str] = []

        # A weak signing key means forgeable tokens, which means no access control at all.
        key = self.secret_key.get_secret_value()
        if self.jwt_algorithm == "HS256":
            if len(key) < 32:
                errors.append("SECRET_KEY must be at least 32 characters")
            if self.is_production and "change-me" in key:
                errors.append("SECRET_KEY is still the example value")
        else:
            for label, path in (
                ("JWT_PRIVATE_KEY_PATH", self.jwt_private_key_path),
                ("JWT_PUBLIC_KEY_PATH", self.jwt_public_key_path),
            ):
                if path is None:
                    errors.append(f"{label} is required when JWT_ALGORITHM=RS256")
                elif not path.is_file():
                    errors.append(f"{label} does not exist: {path}")

        if self.chunk_min_tokens >= self.chunk_max_tokens:
            errors.append("CHUNK_MIN_TOKENS must be below CHUNK_MAX_TOKENS")
        if not (self.chunk_min_tokens <= self.chunk_size <= self.chunk_max_tokens):
            errors.append("CHUNK_SIZE must fall between CHUNK_MIN_TOKENS and CHUNK_MAX_TOKENS")
        if self.context_token_budget >= self.prompt_token_cap:
            errors.append("CONTEXT_TOKEN_BUDGET must be below PROMPT_TOKEN_CAP")

        # A provider selected without its credential fails at first use, in a worker,
        # hours after deploy. Catch it at boot instead.
        required_key = {
            "openai": self.openai_api_key,
            "voyage": self.voyage_api_key,
            "cohere": self.cohere_api_key,
        }.get(self.embedding_provider)
        if required_key is not None and not required_key.get_secret_value():
            errors.append(
                f"EMBEDDING_PROVIDER={self.embedding_provider} requires its API key to be set"
            )
        if self.reranker_provider == "cohere" and not (
            self.cohere_api_key and self.cohere_api_key.get_secret_value()
        ):
            errors.append("RERANKER_PROVIDER=cohere requires COHERE_API_KEY")

        if self.is_production:
            if self.embedding_provider == "fake":
                errors.append("EMBEDDING_PROVIDER=fake is not permitted in production")
            if self.vector_backend == "memory":
                errors.append("VECTOR_BACKEND=memory is not permitted in production")
            if self.storage_backend == "local":
                # Local disk cannot be shared by two API replicas, and a presigned URL
                # served by one of them would 404 on the other.
                errors.append("STORAGE_BACKEND=local is not permitted in production")
            if not self.malware_scan_enabled:
                errors.append("MALWARE_SCAN_ENABLED must be true in production")
            if self.log_format != "json":
                errors.append("LOG_FORMAT must be json in production")
            if not self.rate_limit_enabled:
                errors.append("RATE_LIMIT_ENABLED must be true in production")

        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached because construction reads the filesystem, and because a second instance
    would let two parts of the process disagree about configuration.
    """
    try:
        return Settings()
    except ValueError as exc:  # pragma: no cover - exercised by the boot path, not tests
        sys.stderr.write(f"\nFATAL: {exc}\n\n")
        raise
