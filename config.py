from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """
    Configuration settings for the bot, loaded from environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    bot_token: SecretStr = Field(..., alias="BOT_TOKEN")
    gemini_api_key: SecretStr = Field(..., alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash-lite", alias="GEMINI_MODEL")
    
    # Unified Model Settings (apply to all providers)
    model_temperature: float = Field(0.6, validation_alias="MODEL_TEMPERATURE", alias="GEMINI_TEMPERATURE")
    model_top_p: float = Field(0.95, validation_alias="MODEL_TOP_P", alias="GEMINI_TOP_P")
    model_top_k: int = Field(40, validation_alias="MODEL_TOP_K", alias="GEMINI_TOP_K")
    search_enabled: bool = Field(True, alias="SEARCH_ENABLED")
    
    llm_provider: str = Field("gemini", alias="LLM_PROVIDER")
    groq_api_key: Optional[SecretStr] = Field(None, alias="GROQ_API_KEY")
    groq_model: str = Field("openai/gpt-oss-120b", alias="GROQ_MODEL")
    deepseek_api_key: Optional[SecretStr] = Field(None, alias="DEEPSEEK_API_KEY")
    deepseek_api_model: str = Field("deepseek-chat", alias="DEEPSEEK_API_MODEL")


config = Settings()
