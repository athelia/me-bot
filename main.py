import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from loaders.discord_chat import load_discord_documents
from chroma_client import chroma_client_settings

load_dotenv()

PDF_PATH = Path("data/sample_data.pdf")
DISCORD_DIR = Path("data/discord")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "me-bot"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


def load_documents() -> List[Document]:
    documents: List[Document] = []

    if PDF_PATH.is_file():
        documents.extend(PyPDFLoader(str(PDF_PATH)).load())
        print(f"Loaded PDF pages from {PDF_PATH}")

    discord_documents = load_discord_documents(DISCORD_DIR)
    if discord_documents:
        pairs = sum(1 for d in discord_documents if d.metadata.get("type") == "discord_pair")
        statements = sum(
            1 for d in discord_documents if d.metadata.get("type") == "discord_statement"
        )
        documents.extend(discord_documents)
        print(
            f"Loaded {pairs} conversation pairs and {statements} statements "
            f"from {DISCORD_DIR}"
        )

    print(f"Loaded {len(documents)} documents total")
    return documents


def create_embeddings() -> OllamaEmbeddings:
    """Create a local embedding model via Ollama."""
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def create_vector_store(documents: List[Document]) -> Chroma:
    """Create embeddings and store them in ChromaDB."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = create_embeddings()
    # Warm up the embedding model before bulk work.
    embeddings.embed_query("warmup")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        client_settings=chroma_client_settings(),
    )

    for start in range(0, len(documents), EMBED_BATCH_SIZE):
        batch = documents[start : start + EMBED_BATCH_SIZE]
        vector_store.add_documents(batch)
        done = min(start + EMBED_BATCH_SIZE, len(documents))
        print(f"Embedded {done}/{len(documents)}")

    print(
        f"Created vector store with {vector_store._collection.count()} "
        f"embeddings in {CHROMA_DIR}/"
    )
    return vector_store


if __name__ == "__main__":
    documents = load_documents()
    create_vector_store(documents)
