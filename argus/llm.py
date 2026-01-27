from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from .config import ModelConfig

def get_llm(config: ModelConfig) -> BaseChatModel:
    """
    Factory to return a LangChain chat model based on config.
    """
    if config.provider == "openai":
        return ChatOpenAI(
            model=config.model_name,
            api_key=config.api_key.get_secret_value(),
            temperature=0
        )
    elif config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            api_key=config.api_key.get_secret_value(),
            temperature=0
        )
    elif config.provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.api_key.get_secret_value(),
            temperature=0,
            convert_system_message_to_human=True # Often needed for some Gemini versions
        )
    
    raise ValueError(f"Unsupported provider: {config.provider}")
