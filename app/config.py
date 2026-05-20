from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    model_path: str = "models/anomaly_model.pkl"
    data_path: str = "data/ai4i2020.csv"

    class Config:
        env_file = ".env"


settings = Settings()
