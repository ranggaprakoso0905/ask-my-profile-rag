import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load local .env for local development
load_dotenv(BASE_DIR / ".env")


def get_secret(name: str, default=None):
    """
    Read config from:
    1. Environment variables / .env
    2. Streamlit secrets, if running in Streamlit
    """
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return default


DOCS_DIR = BASE_DIR / "profile_docs"

QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = get_secret(
    "QDRANT_COLLECTION_NAME",
    "ask_my_profile"
)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")