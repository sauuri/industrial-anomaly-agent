from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    model_path: str = "models/anomaly_model.pkl"
    data_path: str = "data/ai4i2020.csv"
    database_url: str = "sqlite:///./anomaly.db"  # Docker에선 .env로 PostgreSQL URL 주입

    class Config:
        env_file = ".env"


settings = Settings()
