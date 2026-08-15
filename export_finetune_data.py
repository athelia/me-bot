"""Export conversation pairs as JSONL for optional LoRA fine-tuning."""

import json
from pathlib import Path

from loaders.discord_chat import load_discord_pairs

DISCORD_DIR = Path("data/discord")
OUTPUT_PATH = Path("output/finetune_pairs.jsonl")


def main() -> None:
    pairs = load_discord_pairs(DISCORD_DIR)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for doc in pairs:
            record = {
                "input": doc.page_content,
                "output": doc.metadata.get("reply", ""),
                "metadata": {
                    "source": doc.metadata.get("source"),
                    "timestamp": doc.metadata.get("timestamp"),
                    "incoming_authors": doc.metadata.get("incoming_authors"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(pairs)} pairs to {OUTPUT_PATH}")
    print("Use this file with unsloth/llama.cpp when ready to fine-tune qwen2.5:7b.")


if __name__ == "__main__":
    main()
