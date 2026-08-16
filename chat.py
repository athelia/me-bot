import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from rich.console import Console
from rich.markdown import Markdown

from chroma_client import chroma_client_settings

load_dotenv()

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "me-bot"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
TOP_K = int(os.getenv("TOP_K", "3"))
STATEMENT_K = int(os.getenv("STATEMENT_K", "3"))
FETCH_K = int(os.getenv("FETCH_K", "12"))
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.88"))
MAX_REPLY_TOKENS = int(os.getenv("MAX_REPLY_TOKENS", "80"))
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "").lower() in {"1", "true", "yes"}
LOG_PATH = Path(os.getenv("CHAT_LOG_PATH", "output/logs.txt"))

console = Console()

_MARKDOWN_NOISE = re.compile(
    r"^#{1,6}\s|^\d+\.\s+\*\*|^\*\*[^*]+\*\*\s*$|^-\s+\*\*",
    re.MULTILINE,
)


def get_vector_store() -> Chroma:
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        client_settings=chroma_client_settings(),
    )


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.35,
        num_predict=MAX_REPLY_TOKENS,
    )


def format_pair_context(docs) -> str:
    seen = set()
    formatted = []
    for doc in docs:
        incoming = doc.page_content.strip()
        reply = doc.metadata.get("reply", "").strip()
        if not incoming or not reply:
            continue
        key = (incoming, reply)
        if key in seen:
            continue
        seen.add(key)
        formatted.append(
            f"They said:\n{incoming}\n\nspacepiratemog replied:\n{reply}"
        )
    return "\n\n---\n\n".join(formatted)


def format_statement_context(docs) -> str:
    seen = set()
    formatted = []
    for doc in docs:
        text = doc.page_content.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        formatted.append(f"spacepiratemog said:\n{text}")
    return "\n\n---\n\n".join(formatted)


def format_context(pairs, statements) -> str:
    sections = []
    pair_text = format_pair_context(pairs)
    statement_text = format_statement_context(statements)
    if pair_text:
        sections.append(
            "Reply examples (how spacepiratemog responds to similar messages):\n"
            + pair_text
        )
    if statement_text:
        sections.append(
            "Topic examples (things spacepiratemog has said):\n" + statement_text
        )
    return "\n\n===\n\n".join(sections)


# Backward compatibility for eval_rag.py
def format_docs(docs) -> str:
    pairs = [d for d in docs if d.metadata.get("type") == "discord_pair"]
    statements = [d for d in docs if d.metadata.get("type") == "discord_statement"]
    if not pairs and not statements and docs:
        pairs = docs
    return format_context(pairs, statements)


PROMPT_WITH_CONTEXT = ChatPromptTemplate.from_messages([
    ("system", """You are spacepiratemog replying in Discord. Write ONLY the next message.

You are NOT an assistant. Do not help, advise, teach, list steps, or explain things unless the examples do.

Hard rules:
- 1-3 sentences max, usually shorter
- Plain text only: no markdown, no headers, no bullet lists, no numbered lists
- Use reply examples for tone and how to respond; use topic examples for what spacepiratemog has said about a subject
- Ignore unrelated details in the examples
- No emojis unless similar examples use them
- Do not invent facts beyond the topic examples
- Never invent dice rolls or game mechanics

{context}"""),
    ("human", "{question}"),
])

PROMPT_NO_CONTEXT = ChatPromptTemplate.from_messages([
    ("system", """You are spacepiratemog in Discord. Write ONE short casual reply (1-2 sentences).

You don't know enough to answer. Do not guess, advise, or be helpful like an AI.
No markdown. No lists. No emojis unless very natural.

Good replies: "idk", "no idea lol", "huh", "maybe?", "Surviving. You?"
Bad replies: advice, explanations, bullet points, or saying 'here are some tips'"""),
    ("human", "{question}"),
])


def sanitize_response(text: str) -> str:
    """Strip assistant-style markdown the model sometimes adds anyway."""
    text = text.strip()
    if _MARKDOWN_NOISE.search(text):
        lines = [
            line for line in text.splitlines()
            if not _MARKDOWN_NOISE.match(line.strip())
        ]
        text = "\n".join(lines).strip()
    # Drop trailing essay paragraphs if model ran away after a good first line
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 2:
        text = "\n\n".join(paragraphs[:2])
    return text.strip()


def _search_by_type(vector_store, question: str, doc_type: str, limit: int):
    scored = vector_store.similarity_search_with_score(
        question,
        k=FETCH_K,
        filter={"type": doc_type},
    )
    filtered = [doc for doc, distance in scored if distance <= MAX_DISTANCE]
    return filtered[:limit]


def get_retrieved_docs(question: str) -> dict:
    vector_store = get_vector_store()
    return {
        "pairs": _search_by_type(vector_store, question, "discord_pair", TOP_K),
        "statements": _search_by_type(
            vector_store, question, "discord_statement", STATEMENT_K
        ),
    }


def generate_reply(question: str) -> str:
    retrieved = get_retrieved_docs(question)
    context = format_context(retrieved["pairs"], retrieved["statements"])
    llm = get_llm()
    if context:
        chain = PROMPT_WITH_CONTEXT | llm | StrOutputParser()
        raw = chain.invoke({"context": context, "question": question})
    else:
        chain = PROMPT_NO_CONTEXT | llm | StrOutputParser()
        raw = chain.invoke({"question": question})
    return sanitize_response(raw)


def append_chat_log(user_message: str, bot_response: str) -> None:
    """Append one exchange to the chat log file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"[{timestamp}]\n"
        f"You: {user_message}\n"
        f"Mog: {bot_response}\n"
        f"{'-' * 60}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)


def build_rag_chain():
    """Compatibility wrapper for eval_rag.py."""
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(generate_reply)


def main() -> None:
    console.print("\n[bold green]RAG Chatbot Ready[/bold green]")
    console.print("Type your messages below. Type 'quit' to exit.\n")

    while True:
        question = console.input("[bold cyan]You:[/bold cyan] ")
        if question.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        console.print("\n[bold green]Mog:[/bold green]")
        if SHOW_SOURCES:
            retrieved = get_retrieved_docs(question)
            pairs = retrieved["pairs"]
            statements = retrieved["statements"]
            if pairs:
                console.print("[dim]Retrieved pairs:[/dim]")
                for i, doc in enumerate(pairs, 1):
                    incoming = doc.page_content.strip().replace("\n", " ")[:80]
                    reply = doc.metadata.get("reply", "").replace("\n", " ")[:80]
                    console.print(f"[dim]  {i}. They: {incoming}[/dim]")
                    console.print(f"[dim]     Mog: {reply}[/dim]")
            if statements:
                console.print("[dim]Retrieved statements:[/dim]")
                for i, doc in enumerate(statements, 1):
                    text = doc.page_content.strip().replace("\n", " ")[:120]
                    console.print(f"[dim]  {i}. {text}[/dim]")
            if not pairs and not statements:
                console.print(
                    "[dim]No results passed similarity threshold — using short fallback.[/dim]"
                )
            console.print()

        response = generate_reply(question)
        console.print(Markdown(response))
        console.print()
        append_chat_log(question, response)


if __name__ == "__main__":
    main()
