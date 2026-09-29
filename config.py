"""
Centralized configuration for the IT Support RAG Agent.

Single source of truth for deployment-tunable values. Defaults select the
documented models, tracked index, and feature flags. Environment variables
can adjust models, thresholds, retry behavior, and optional features.
"""

import os
from pathlib import Path
from index_contract import INDEX_EMBEDDING_MODEL


PROJECT_DIR = Path(__file__).resolve().parent


def resolve_chroma_path(value: str | None = None) -> str:
    """Resolve relative index paths from the repository, regardless of cwd."""
    configured = value if value is not None else os.getenv("CHROMA_DB_PATH", "chroma_db")
    if not configured.strip():
        raise ValueError("CHROMA_DB_PATH must not be empty")
    return str((PROJECT_DIR / Path(configured).expanduser()).resolve())


def _get_env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


class Config:
    CHROMA_DB_PATH = resolve_chroma_path()

    # Model selection
    CHAT_MODEL = _get_env_str(
        "CHAT_MODEL", "gpt-4o-mini"
    )
    EMBEDDING_MODEL = _get_env_str(
        "EMBEDDING_MODEL", INDEX_EMBEDDING_MODEL
    )

    # Fine-tuned model wiring — unused until a model exists. When
    # USE_FINETUNED_MODEL is true AND FINETUNED_MODEL_ID is set, the pipeline
    # serves that model instead of CHAT_MODEL. Defaults keep the base model, so
    # the default serving model remains selected until both env vars are set.
    FINETUNED_MODEL_ID = _get_env_str(
        "FINETUNED_MODEL_ID", ""
    )
    USE_FINETUNED_MODEL = _get_env_bool(
        "USE_FINETUNED_MODEL", False
    )

    # Retrieval
    RETRIEVAL_K = _get_env_int("RETRIEVAL_K", 3)

    # Retrieval review uses the raw 0.20 relevance-score cutoff. The 0.60
    # legacy display setting remains configurable but is not used by chat.
    # Neither value is a calibrated probability.
    CONFIDENCE_HIGH_THRESHOLD = _get_env_float(
        "CONFIDENCE_HIGH_THRESHOLD", 0.60
    )
    CONFIDENCE_MEDIUM_THRESHOLD = _get_env_float(
        "CONFIDENCE_MEDIUM_THRESHOLD", 0.20
    )

    # Retry behavior
    MAX_RETRIEVAL_RETRIES = _get_env_int(
        "MAX_RETRIEVAL_RETRIES", 2
    )
    RETRY_DELAY_SECONDS = _get_env_float(
        "RETRY_DELAY_SECONDS", 1.0
    )
    LANGGRAPH_RECURSION_LIMIT = _get_env_int(
        "LANGGRAPH_RECURSION_LIMIT", 20
    )

    # Input validation
    MAX_QUESTION_LENGTH = _get_env_int(
        "MAX_QUESTION_LENGTH", 2000
    )

    # Feature flags - allow disabling features without
    # code changes, useful for debugging or gradual rollout
    ENABLE_LIVE_EVAL = _get_env_bool(
        "ENABLE_LIVE_EVAL", True
    )
    ENABLE_VERIFICATION_AGENT = _get_env_bool(
        "ENABLE_VERIFICATION_AGENT", True
    )
    ENABLE_ADAPTIVE_RETRY = _get_env_bool(
        "ENABLE_ADAPTIVE_RETRY", True
    )
    ENABLE_SERVICENOW_LIVE = _get_env_bool(
        "ENABLE_SERVICENOW_LIVE", True
    )
    ENABLE_IMAGE_UPLOAD = _get_env_bool(
        "ENABLE_IMAGE_UPLOAD", True
    )

    # Rate limiting (per session). Generous defaults so normal testing never
    # trips it; blocks obvious abuse before it can incur API cost.
    RATE_LIMIT_MAX_REQUESTS = _get_env_int(
        "RATE_LIMIT_MAX_REQUESTS", 15
    )
    RATE_LIMIT_WINDOW_SECONDS = _get_env_int(
        "RATE_LIMIT_WINDOW_SECONDS", 60
    )
    ENABLE_RATE_LIMITING = _get_env_bool(
        "ENABLE_RATE_LIMITING", True
    )

    # Environment identification
    ENVIRONMENT = _get_env_str(
        "APP_ENVIRONMENT", "production"
    )

    @classmethod
    def active_chat_model(cls) -> str:
        """
        The chat model to actually serve: the fine-tuned model when it is both
        explicitly enabled AND an ID is present, otherwise the base CHAT_MODEL.
        Reads env live so the switch can be flipped without a code change; the
        default path (USE_FINETUNED_MODEL unset/false, or no ID) returns
        CHAT_MODEL. This configuration alone does not indicate that a
        fine-tuned model exists or has been evaluated.
        """
        use_ft = _get_env_bool(
            "USE_FINETUNED_MODEL", cls.USE_FINETUNED_MODEL
        )
        ft_id = _get_env_str(
            "FINETUNED_MODEL_ID", cls.FINETUNED_MODEL_ID
        ).strip()
        if use_ft and ft_id:
            return ft_id
        return cls.CHAT_MODEL

    @classmethod
    def summary(cls) -> dict:
        """Returns current effective config for
        debugging/display purposes."""
        return {
            "chat_model": cls.CHAT_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "retrieval_k": cls.RETRIEVAL_K,
            "confidence_high":
                cls.CONFIDENCE_HIGH_THRESHOLD,
            "confidence_medium":
                cls.CONFIDENCE_MEDIUM_THRESHOLD,
            "max_retries": cls.MAX_RETRIEVAL_RETRIES,
            "max_question_length":
                cls.MAX_QUESTION_LENGTH,
            "live_eval_enabled": cls.ENABLE_LIVE_EVAL,
            "verification_enabled":
                cls.ENABLE_VERIFICATION_AGENT,
            "adaptive_retry_enabled":
                cls.ENABLE_ADAPTIVE_RETRY,
            "servicenow_live_enabled":
                cls.ENABLE_SERVICENOW_LIVE,
            "image_upload_enabled":
                cls.ENABLE_IMAGE_UPLOAD,
            "rate_limiting_enabled":
                cls.ENABLE_RATE_LIMITING,
            "environment": cls.ENVIRONMENT,
        }


config = Config()
