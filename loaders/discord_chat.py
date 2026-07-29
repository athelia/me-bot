import re
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

TARGET_AUTHOR = "spacepiratemog"

# Discord plain-text export: "author — M/D/YYYY H:MM AM/PM"
_MESSAGE_HEADER = re.compile(
    r"^(.+?)\s*[—–-]\s*"
    r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+(?:AM|PM))\s*$",
    re.IGNORECASE,
)


def _normalize_author(author: str) -> str:
    return author.strip().rstrip(",")


def _is_target_author(author: str) -> bool:
    return _normalize_author(author).casefold() == TARGET_AUTHOR.casefold()


def parse_discord_export(text: str, *, source: str = "") -> List[Document]:
    """Parse a Discord plain-text export, keeping only TARGET_AUTHOR messages."""
    documents: List[Document] = []
    current_author: Optional[str] = None
    current_timestamp: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_author, current_timestamp, current_lines
        if not current_author or not current_lines:
            current_author = None
            current_timestamp = None
            current_lines = []
            return
        if _is_target_author(current_author):
            documents.append(
                Document(
                    page_content="\n".join(current_lines).strip(),
                    metadata={
                        "source": source,
                        "author": _normalize_author(current_author),
                        "timestamp": current_timestamp,
                        "type": "discord_message",
                    },
                )
            )
        current_author = None
        current_timestamp = None
        current_lines = []

    for line in text.splitlines():
        header = _MESSAGE_HEADER.match(line)
        if header:
            flush()
            current_author = header.group(1)
            current_timestamp = header.group(2)
            continue
        if current_author is not None:
            current_lines.append(line)

    flush()
    return documents


def load_discord_chats(
    directory: Path,
    *,
    extensions: Iterable[str] = (".txt",),
) -> List[Document]:
    """Load Discord exports from a directory, one document per TARGET_AUTHOR message."""
    if not directory.is_dir():
        return []

    documents: List[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.extend(parse_discord_export(text, source=str(path)))

    return documents
