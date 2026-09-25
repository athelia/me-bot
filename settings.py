import os
from pathlib import Path
import dotenv
dotenv.load_dotenv()

PERSONA=os.getenv("PERSONA", {"screenname":"spacepiratemog", "displayname":"Mog"})



DISCORD_DIR = Path("data/discord")
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "me-bot"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
TOP_K = int(os.getenv("TOP_K", "3"))
STATEMENT_K = int(os.getenv("STATEMENT_K", "3"))
FETCH_K = int(os.getenv("FETCH_K", "12"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.88"))
MAX_REPLY_TOKENS = int(os.getenv("MAX_REPLY_TOKENS", "80"))
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "").lower() in {"1", "true", "yes"}
LOG_PATH = Path(os.getenv("CHAT_LOG_PATH", "output/logs.txt"))