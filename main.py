import os
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from loaders.discord_chat import load_discord_pairs

load_dotenv()

PDF_PATH = Path("data/sample_data.pdf")
DISCORD_DIR = Path("data/discord")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "me-bot"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")


def load_documents() -> List[Document]:
    documents: List[Document] = []

    if PDF_PATH.is_file():
        documents.extend(PyPDFLoader(str(PDF_PATH)).load())
        print(f"Loaded PDF pages from {PDF_PATH}")

    pair_documents = load_discord_pairs(DISCORD_DIR)
    if pair_documents:
        documents.extend(pair_documents)
        print(f"Loaded {len(pair_documents)} Discord conversation pairs from {DISCORD_DIR}")

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
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(
        f"Created vector store with {vector_store._collection.count()} "
        f"embeddings in {CHROMA_DIR}/"
    )
    return vector_store


if __name__ == "__main__":
    documents = load_documents()
    create_vector_store(documents)
