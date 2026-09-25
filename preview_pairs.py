from pathlib import Path

from loaders.discord_chat import load_discord_pairs
from settings import PERSONA, DISCORD_DIR

OUTPUT_PATH = Path("output/discord_pairs_sample.txt")
SAMPLE_COUNT = 10


def format_pair(index: int, total: int, doc) -> str:
    lines = [
        f"[{index}/{total}] {doc.metadata.get('timestamp', 'unknown time')}",
        f"source: {doc.metadata.get('source', 'unknown')}",
        f"from: {doc.metadata.get('incoming_authors', 'unknown')}",
        "",
        "When someone said:",
        doc.page_content,
        "",
        f"{PERSONA['displayname']} replied:",
        doc.metadata.get("reply", ""),
        "",
    ]
    return "\n".join(lines)


def build_preview(documents, sample_count: int = SAMPLE_COUNT) -> str:
    total = len(documents)
    if total == 0:
        return "No conversation pairs found.\n"

    sample_size = min(sample_count, total)
    if total <= sample_size:
        samples = documents
        header = f"Showing all {total} pairs."
    else:
        half = sample_size // 2
        samples = documents[:half] + documents[-(sample_size - half) :]
        header = f"Showing {sample_size} samples (first {half} and last {sample_size - half})."

    sections = [
        "Discord conversation pair preview",
        f"Target author: {PERSONA['screenname']}",
        f"Total pairs: {total}",
        header,
        "=" * 60,
        "",
    ]

    for i, doc in enumerate(samples, start=1):
        sections.append(format_pair(i, sample_size, doc))
        sections.append("-" * 60)
        sections.append("")

    return "\n".join(sections)


def main() -> None:
    documents = load_discord_pairs(DISCORD_DIR)
    preview = build_preview(documents)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(preview, encoding="utf-8")

    print(f"Loaded {len(documents)} conversation pairs")
    print(f"Wrote preview to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
