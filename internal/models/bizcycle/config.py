import os
from pathlib import Path
from urllib.parse import quote_plus

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for local runs
    load_dotenv = None


def _load_env() -> None:
    """Load environment variables from a local .env if available.

    This keeps the code portable by avoiding hard-coded, platform-specific paths.
    """

    if load_dotenv is None:
        return

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / "internal" / ".env"
    load_dotenv(dotenv_path=env_path)


_load_env()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

if all([DB_USER, DB_NAME]):
    DATABASE_URL = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
else:
    # Leave empty so callers can fall back to local CSV data when a DB is absent.
    DATABASE_URL = None


def get_database_url():
    return DATABASE_URL
