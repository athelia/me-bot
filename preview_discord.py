from pathlib import Path

from loaders.discord_chat import load_discord_chats

DISCORD_DIR = Path("data/discord")
OUTPUT_PATH = Path("output/discord_sample.txt")
SAMPLE_COUNT = 10


def format_message(index: int, total: int, doc) -> str:
    lines = [
        f"[{index}/{total}] {doc.metadata.get('timestamp', 'unknown time')}",
        f"source: {doc.metadata.get('source', 'unknown')}",
        doc.page_content,
        "",
    ]
    return "\n".join(lines)


def build_preview(documents, sample_count: int = SAMPLE_COUNT) -> str:
    total = len(documents)
    if total == 0:
        return "No Discord messages found.\n"

    sample_size = min(sample_count, total)
    if total <= sample_size:
        samples = documents
        header = f"Showing all {total} messages."
    else:
        half = sample_size // 2
        first = documents[:half]
        last = documents[-(sample_size - half) :]
        samples = first + last
        header = f"Showing {sample_size} samples (first {half} and last {sample_size - half})."

    sections = [
        "Discord sample preview",
        f"Target author: spacepiratemog",
        f"Total messages: {total}",
        header,
        "=" * 60,
        "",
    ]

    for i, doc in enumerate(samples, start=1):
        sections.append(format_message(i, sample_size, doc))
        sections.append("-" * 60)
        sections.append("")

    return "\n".join(sections)


def main() -> None:
    documents = load_discord_chats(DISCORD_DIR)
    preview = build_preview(documents)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(preview, encoding="utf-8")

    print(f"Loaded {len(documents)} messages")
    print(f"Wrote preview to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
