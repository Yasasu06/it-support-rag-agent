"""
Centralized configuration for the IT Support RAG Agent.

Single source of truth for deployment-tunable values. Every setting defaults to
EXACTLY what the system used before this module existed, so importing config
changes nothing unless an environment variable is explicitly set. This lets a
deployment tune models, thresholds, retry behavior, and feature flags without
touching code.
"""

import os


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
    # Model selection
    CHAT_MODEL = _get_env_str(
        "CHAT_MODEL", "gpt-4o-mini"
    )
    EMBEDDING_MODEL = _get_env_str(
        "EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # Retrieval
    RETRIEVAL_K = _get_env_int("RETRIEVAL_K", 3)

    # Confidence thresholds (must match existing
    # calibrated values exactly - verify against
    # confidence_from_score in app.py before finalizing)
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

    # Environment identification
    ENVIRONMENT = _get_env_str(
        "APP_ENVIRONMENT", "production"
    )

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
            "environment": cls.ENVIRONMENT,
        }


config = Config()
