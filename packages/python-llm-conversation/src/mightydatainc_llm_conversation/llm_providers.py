GPT_MODEL_CHEAP = "gpt-5.6-luna"
GPT_MODEL_STANDARD = "gpt-6-luna"
GPT_MODEL_SMART = "gpt-6-luna"
GPT_MODEL_VISION = "gpt-6-luna"

CLAUDE_MODEL_CHEAP = "claude-haiku-4-5-20251001"
CLAUDE_MODEL_STANDARD = "claude-sonnet-4-6"
CLAUDE_MODEL_SMART = "claude-opus-4-6"
CLAUDE_MODEL_VISION = "claude-opus-4-6"

LLM_RETRY_LIMIT_DEFAULT = 5
LLM_RETRY_BACKOFF_TIME_SECONDS_DEFAULT = 30


def identify_llm_provider(ai_client) -> str:
    """Identify provider from client capabilities (messages => anthropic)."""
    if hasattr(ai_client, "messages") and getattr(ai_client, "messages") is not None:
        return "anthropic"
    return "openai"


def get_model_name(provider: str, tier: str) -> str:
    """Resolve provider+tier to a concrete model name."""
    if provider == "openai":
        if tier == "cheap":
            return GPT_MODEL_CHEAP
        if tier == "standard":
            return GPT_MODEL_STANDARD
        if tier == "smart":
            return GPT_MODEL_SMART
        if tier == "vision":
            return GPT_MODEL_VISION

    if provider == "anthropic":
        if tier == "cheap":
            return CLAUDE_MODEL_CHEAP
        if tier == "standard":
            return CLAUDE_MODEL_STANDARD
        if tier == "smart":
            return CLAUDE_MODEL_SMART
        if tier == "vision":
            return CLAUDE_MODEL_VISION

    raise ValueError(f"Unsupported provider or tier: {provider}, {tier}")
