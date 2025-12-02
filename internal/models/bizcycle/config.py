import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load the .env from the project root
load_dotenv(dotenv_path="C:/Users/harte/Documents/goTutorial/internal/.env")
# Pull environment variables
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD"))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# Build SQLAlchemy Postgres URI
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def get_database_url():
    return DATABASE_URL
