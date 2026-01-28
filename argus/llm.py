from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from .config import ModelConfig


def _build_common_kwargs(config: ModelConfig) -> Dict[str, Any]:
    """
    Map generic ModelConfig knobs to provider-specific kwargs as safely as possible.
    """
    kwargs: Dict[str, Any] = {
        "model": config.model_name,
        "temperature": config.temperature,
    }

    # Token / timeout controls – only attached where supported.
    if config.max_output_tokens is not None:
        # OpenAI / Anthropic use `max_tokens`, Google uses `max_output_tokens`.
        # We attach provider-specific keys in get_llm; here we expose a generic value.
        kwargs["max_tokens"] = config.max_output_tokens

    if config.timeout_s is not None:
        kwargs["timeout"] = config.timeout_s

    # Allow provider-specific overrides via extra
    kwargs.update(config.extra or {})
    return kwargs


def get_llm(config: ModelConfig) -> BaseChatModel:
    """
    Factory to return a LangChain chat model based on config.
    """
    common_kwargs = _build_common_kwargs(config)

    if config.provider == "openai":
        return ChatOpenAI(
            api_key=config.api_key.get_secret_value(),
            **common_kwargs,
        )
    elif config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Anthropic's Chat wrapper uses a similar signature to OpenAI in langchain.
        return ChatAnthropic(
            api_key=config.api_key.get_secret_value(),
            **common_kwargs,
        )
    elif config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Adjust token key for Gemini/Google wrapper if caller didn't override via `extra`.
        google_kwargs: Dict[str, Any] = dict(common_kwargs)
        # `model` will be passed explicitly below, so avoid passing it twice.
        google_kwargs.pop("model", None)
        if "max_output_tokens" not in google_kwargs and "max_tokens" in google_kwargs:
            google_kwargs["max_output_tokens"] = google_kwargs.pop("max_tokens")

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.api_key.get_secret_value(),
            convert_system_message_to_human=True,  # Often needed for some Gemini versions
            **google_kwargs,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
