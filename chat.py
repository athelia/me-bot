import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "me-bot"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
TOP_K = 5
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "").lower() in {"1", "true", "yes"}

console = Console()


def get_vector_store() -> Chroma:
    """Connect to the existing ChromaDB vector store."""
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string."""
    seen = set()
    formatted = []
    for doc in docs:
        content = doc.page_content.strip()
        if not content or content in seen:
            continue
        seen.add(content)
        formatted.append(content)
    return "\n\n---\n\n".join(formatted)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are spacepiratemog in a Discord chat. Reply as this person, not as an AI assistant.

Use ONLY the past messages below as your voice and knowledge. Do not invent facts, people, or backstory.

Style rules:
- Write like casual Discord: short, direct, conversational
- Match the tone of the examples (plain, sometimes blunt, sometimes warm)
- Do not sound polite, formal, or customer-servicey
- Do not use emojis unless the examples below use them for a similar kind of reply
- Keep answers brief unless the question needs more detail
- If the examples do not mention something, say "I don't know" or "no idea" — do not guess

Past messages from spacepiratemog:
{context}"""),
    ("human", "{question}"),
])


def get_retrieved_docs(question: str):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20},
    )
    return retriever.invoke(question)


def build_rag_chain():
    """Build the complete RAG chain."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20},
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.6,
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def main() -> None:
    """Run the interactive chatbot."""
    console.print("\n[bold green]RAG Chatbot Ready[/bold green]")
    console.print("Type your questions below. Type 'quit' to exit.\n")

    chain = build_rag_chain()

    while True:
        question = console.input("[bold cyan]You:[/bold cyan] ")
        if question.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        console.print("\n[bold green]Mog:[/bold green]")
        if SHOW_SOURCES:
            docs = get_retrieved_docs(question)
            console.print("[dim]Retrieved messages:[/dim]")
            for i, doc in enumerate(docs, 1):
                preview = doc.page_content.strip().replace("\n", " ")[:120]
                console.print(f"[dim]  {i}. {preview}[/dim]")
            console.print()
        response = chain.invoke(question)
        console.print(Markdown(response))
        console.print()


if __name__ == "__main__":
    main()
