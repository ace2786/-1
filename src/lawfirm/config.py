"""Global settings via pydantic-settings; env prefix LAWFIRM_."""
from pathlib import Path
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


class Settings(BaseSettings):
    # model routing (dual-model per spec)
    heavy_model: str = "qwen3:8b"          # deep case-file analysis
    light_model: str = "qwen3:1.7b"        # fast chat/UI interaction
    ollama_base_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text:v1.5"

    # storage
    cases_dir: Path = DATA / "cases"
    audit_dir: Path = DATA / "audit"
    logs_dir: Path = DATA / "logs"
    flywheel_dir: Path = DATA / "flywheel"
    chroma_dir: Path = DATA / "chroma"

    # limits & safety
    max_upload_mb: int = 50
    allowed_exts: set[str] = {".pdf", ".txt", ".md", ".xlsx", ".xls", ".png", ".jpg", ".jpeg"}
    bind_host: str = "127.0.0.1"           # local-only by default (security)
    api_token: str = ""                    # if set, require X-API-Token header

    # observability
    log_level: str = "INFO"

    model_config = {
        "env_prefix": "LAWFIRM_",
        "extra": "ignore",
    }


_settings_file = Path(__file__).resolve().parents[2] / "lawfirm.local.env"
if _settings_file.exists():
    from dotenv import load_dotenv  # optional; provided by pydantic-settings deps
    load_dotenv(_settings_file)

settings = Settings()
settings.cases_dir.mkdir(parents=True, exist_ok=True)
settings.audit_dir.mkdir(parents=True, exist_ok=True)
settings.logs_dir.mkdir(parents=True, exist_ok=True)
settings.flywheel_dir.mkdir(parents=True, exist_ok=True)