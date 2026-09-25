"""Run fixed prompts through retrieval and optionally the full RAG chain."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from chat import build_rag_chain, format_context, get_retrieved_docs
from settings import PERSONA
    
load_dotenv()

DEFAULT_PROMPTS = [
    "how are you?",
    "who are you?",
    "want to watch something?",
    "who is Lily?",
    "what do you think about Loki?",
    "Hello, how are you?",
    "Are you going to the party tomorrow?",
    "Brainweasel check in: we're good right?",
]

OUTPUT_PATH = Path("output/eval_results.txt")


def run_eval(prompts: list[str], *, generate: bool = False) -> str:
    sections = ["RAG evaluation", f"Prompts: {len(prompts)}", "=" * 60, ""]

    chain = build_rag_chain() if generate else None

    for prompt in prompts:
        sections.append(f"## Prompt: {prompt}")
        sections.append("")

        retrieved = get_retrieved_docs(prompt)
        pairs = retrieved["pairs"]
        statements = retrieved["statements"]

        sections.append("### Retrieved pairs")
        if not pairs:
            sections.append("(none)")
        for i, doc in enumerate(pairs, 1):
            sections.append(f"{i}. When someone said:")
            sections.append(doc.page_content)
            sections.append(f"   {PERSONA['displayname']} replied: {doc.metadata.get('reply', '')}")
            sections.append("")

        sections.append("### Retrieved statements")
        if not statements:
            sections.append("(none)")
        for i, doc in enumerate(statements, 1):
            sections.append(f"{i}. {PERSONA['displayname']} said:")
            sections.append(doc.page_content)
            sections.append("")

        sections.append("### Formatted context")
        sections.append(format_context(pairs, statements) or "(empty)")
        sections.append("")

        if chain:
            sections.append("### Generated reply")
            sections.append(chain.invoke(prompt))
            sections.append("")

        sections.append("-" * 60)
        sections.append("")

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and replies.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also run the LLM and include generated replies (slower).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Additional prompt to evaluate (can repeat).",
    )
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    report = run_eval(prompts, generate=args.generate)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(f"Wrote evaluation to {OUTPUT_PATH}")
    if not args.generate:
        print("Retrieval only. Pass --generate to include LLM replies.")


if __name__ == "__main__":
    main()
