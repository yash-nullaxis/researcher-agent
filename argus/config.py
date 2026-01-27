from pydantic import BaseModel, Field, SecretStr

class ModelConfig(BaseModel):
    provider: str = "openai"  # openai, anthropic, azure
    model_name: str = "gpt-4-turbo"
    api_key: SecretStr
    api_base: str = None
    
class AgentConfig(BaseModel):
    model: ModelConfig
    db_connection_str: str
    max_steps: int = 15
    recursion_limit: int = 25
    verbose: bool = False
