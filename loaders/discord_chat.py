import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from langchain_core.documents import Document
from settings import PERSONA

TARGET_AUTHOR = PERSONA["screenname"]
MIN_REPLY_CHARS = 3

_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_MESSAGE_HEADER = re.compile(
    r"^(.+?)\s*[—–-]\s*"
    r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+(?:AM|PM))"
    r"(?:[A-Za-z]+day,.*?)?\s*$",
    re.IGNORECASE,
)
_EMBED_BOILERPLATE = {
    "image",
    "youtube",
    "teleparty",
    "watch tv together with your friends on teleparty!",
    "attachment",
}
_DISCORD_UI_LINES = {
    "click to react",
    "add reaction",
    "reply",
    "forward",
    "more",
    "edit",
    "jump",
    "mark unread",
    "pin message",
    "create thread",
}
_EMOJI_SHORTCODE = re.compile(r"^:[a-z0-9_+-]+:$", re.IGNORECASE)
_STANDALONE_DATE = re.compile(
    r"^[A-Za-z]+ \d{1,2}, \d{4}$"
)


@dataclass
class ParsedMessage:
    author: str
    timestamp: str
    content: str


def _normalize_author(author: str) -> str:
    return author.strip().rstrip(",")


def _is_target_author(author: str) -> bool:
    return _normalize_author(author).casefold() == TARGET_AUTHOR.casefold()


def _clean_line(line: str) -> str:
    line = line.replace("\xa0", " ").strip()
    line = _URL_PATTERN.sub("", line).strip()
    lowered = line.casefold()
    if lowered in _EMBED_BOILERPLATE or lowered in _DISCORD_UI_LINES:
        return ""
    if _EMOJI_SHORTCODE.match(line):
        return ""
    if _STANDALONE_DATE.match(line):
        return ""
    if lowered.startswith("started a call"):
        return ""
    if "server tag:" in lowered:
        return ""
    return line


def clean_message_content(content: str) -> str:
    """Remove URLs, embed boilerplate, and empty lines from a message."""
    lines = [_clean_line(line) for line in content.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def is_low_signal_content(content: str) -> bool:
    """True for link/image-only or very short low-information messages."""
    cleaned = clean_message_content(content)
    if not cleaned:
        return True
    if cleaned.casefold() in _EMBED_BOILERPLATE:
        return True
    if len(re.sub(r"[\W_]+", "", cleaned, flags=re.UNICODE)) < MIN_REPLY_CHARS:
        return True
    return False


def _parse_all_messages(text: str) -> List[ParsedMessage]:
    messages: List[ParsedMessage] = []
    current_author: Optional[str] = None
    current_timestamp: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_author, current_timestamp, current_lines
        if not current_author:
            current_author = None
            current_timestamp = None
            current_lines = []
            return
        content = clean_message_content("\n".join(current_lines))
        if content:
            messages.append(
                ParsedMessage(
                    author=_normalize_author(current_author),
                    timestamp=current_timestamp or "",
                    content=content,
                )
            )
        current_author = None
        current_timestamp = None
        current_lines = []

    for line in text.splitlines():
        line = line.replace("\xa0", " ")
        header = _MESSAGE_HEADER.match(line)
        if header:
            flush()
            current_author = header.group(1)
            current_timestamp = header.group(2)
            continue
        if current_author is not None:
            current_lines.append(line)

    flush()
    return messages


def _format_incoming(messages: List[ParsedMessage]) -> tuple[str, str]:
    parts: List[str] = []
    authors: List[str] = []
    for msg in messages:
        cleaned = clean_message_content(msg.content)
        if cleaned and not is_low_signal_content(cleaned):
            parts.append(f"{msg.author}: {cleaned}")
            authors.append(msg.author)
    incoming_text = "\n".join(parts)
    unique_authors = ", ".join(dict.fromkeys(authors))
    return incoming_text, unique_authors


def parse_discord_conversation_pairs(text: str, *, source: str = "") -> List[Document]:
    """Build prompt/response pairs: incoming context -> spacepiratemog reply."""
    documents: List[Document] = []
    pending_incoming: List[ParsedMessage] = []

    for msg in _parse_all_messages(text):
        if _is_target_author(msg.author):
            reply = clean_message_content(msg.content)
            incoming_snapshot = list(pending_incoming)
            pending_incoming = []

            if is_low_signal_content(reply):
                continue

            incoming_text, incoming_authors = _format_incoming(incoming_snapshot)
            if not incoming_text:
                continue

            documents.append(
                Document(
                    page_content=incoming_text,
                    metadata={
                        "source": source,
                        "reply": reply,
                        "timestamp": msg.timestamp,
                        "type": "discord_pair",
                        "incoming_authors": incoming_authors,
                    },
                )
            )
        else:
            pending_incoming.append(msg)

    return documents


def parse_discord_statements(text: str, *, source: str = "") -> List[Document]:
    """Index every spacepiratemog message for topic/opinion retrieval."""
    documents: List[Document] = []
    for msg in _parse_all_messages(text):
        if not _is_target_author(msg.author):
            continue
        if is_low_signal_content(msg.content):
            continue
        documents.append(
            Document(
                page_content=msg.content,
                metadata={
                    "source": source,
                    "author": msg.author,
                    "timestamp": msg.timestamp,
                    "type": "discord_statement",
                },
            )
        )
    return documents


def load_discord_statements(
    directory: Path,
    *,
    extensions: Iterable[str] = (".txt",),
) -> List[Document]:
    """Load spacepiratemog statements for topic retrieval."""
    if not directory.is_dir():
        return []

    documents: List[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.extend(parse_discord_statements(text, source=str(path)))

    return documents


def load_discord_documents(
    directory: Path,
    *,
    extensions: Iterable[str] = (".txt",),
) -> List[Document]:
    """Load conversation pairs and statements for dual-index RAG."""
    return load_discord_pairs(directory, extensions=extensions) + load_discord_statements(
        directory, extensions=extensions
    )


def load_discord_pairs(
    directory: Path,
    *,
    extensions: Iterable[str] = (".txt",),
) -> List[Document]:
    """Load conversation pairs from Discord exports for RAG retrieval."""
    if not directory.is_dir():
        return []

    documents: List[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.extend(parse_discord_conversation_pairs(text, source=str(path)))

    return documents