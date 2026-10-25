from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    database_url: str = f"sqlite+aiosqlite:///{BASE_DIR}/netguard.db"

    rf_model_path: Path = BASE_DIR / "models" / "rf_cicids.joblib"
    ae_model_path: Path = BASE_DIR / "models" / "lstm_ae_cicids.pt"
    scaler_path: Path = BASE_DIR / "models" / "scaler.joblib"
    selector_path: Path = BASE_DIR / "models" / "selector.joblib"

    ae_threshold: float = 0.023
    replay_speed_ms: int = 200
    mcp_port: int = 8001

    # CIC-IDS2017 feature names (79 features, whitespace-stripped)
    feature_names: list[str] = []

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
