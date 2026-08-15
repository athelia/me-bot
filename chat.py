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

load_dotenv()

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "me-bot"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
TOP_K = int(os.getenv("TOP_K", "3"))
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
    )


def get_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.35,
        num_predict=MAX_REPLY_TOKENS,
    )


def format_docs(docs) -> str:
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


PROMPT_WITH_CONTEXT = ChatPromptTemplate.from_messages([
    ("system", """You are spacepiratemog replying in Discord. Write ONLY the next message.

You are NOT an assistant. Do not help, advise, teach, list steps, or explain things unless the examples do.

Hard rules:
- 1-3 sentences max, usually shorter
- Plain text only: no markdown, no headers, no bullet lists, no numbered lists
- Match the casual tone and length of spacepiratemog's replies in the examples
- Ignore unrelated details in the examples — copy vibe, not content
- Do not mention topics from the examples unless the user brought them up
- No emojis unless the examples use them for a similar reply
- Do not invent facts

Examples of how spacepiratemog replies to similar messages:
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


def get_retrieved_docs(question: str):
    vector_store = get_vector_store()
    scored = vector_store.similarity_search_with_score(question, k=FETCH_K)
    filtered = [doc for doc, distance in scored if distance <= MAX_DISTANCE]
    return filtered[:TOP_K]


def generate_reply(question: str) -> str:
    docs = get_retrieved_docs(question)
    context = format_docs(docs)
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
            docs = get_retrieved_docs(question)
            if docs:
                console.print("[dim]Retrieved pairs:[/dim]")
                for i, doc in enumerate(docs, 1):
                    incoming = doc.page_content.strip().replace("\n", " ")[:80]
                    reply = doc.metadata.get("reply", "").replace("\n", " ")[:80]
                    console.print(f"[dim]  {i}. They: {incoming}[/dim]")
                    console.print(f"[dim]     Mog: {reply}[/dim]")
            else:
                console.print("[dim]No pairs passed similarity threshold — using short fallback.[/dim]")
            console.print()

        response = generate_reply(question)
        console.print(Markdown(response))
        console.print()
        append_chat_log(question, response)


if __name__ == "__main__":
    main()
