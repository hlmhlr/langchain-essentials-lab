"""Extension A: ask questions about your own documents in the terminal.

Aim
    Use the RAG pipeline from the lab as a real tool: point it at a folder,
    ask questions, get answers with sources, and ask follow-ups.

What you'll see
    The indexed files and chunk count, then a prompt. Each answer is
    followed by its numbered sources (file, page) and, for follow-ups,
    the standalone question that was actually searched.

What to notice
    - The first run embeds everything (a few seconds); later runs reuse the
      saved index in chroma_db/ unless files changed.
    - Ask something the documents don't cover: the answer should be
      "I don't know based on the documents."
    - Ask a follow-up like "what about its memory?" and check the rewritten
      question.

Run:  uv run python scripts/18_ask_docs.py                        (data/downloads)
      uv run python scripts/18_ask_docs.py --folder path\\to\\docs  (your own files)
      uv run python scripts/18_ask_docs.py --rebuild               (re-index)
"""

import argparse
from pathlib import Path

from lclab.doc_qa import DEFAULT_FOLDER, DocQA
from lclab.llm import get_embeddings, get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="folder with .pdf/.html/.txt/.md files")
    parser.add_argument("--rebuild", action="store_true", help="re-index even if nothing changed")
    args = parser.parse_args()

    qa = DocQA(get_llm(), get_embeddings(), folder=args.folder, rebuild=args.rebuild)
    print(f"{'Indexed' if qa.rebuilt else 'Loaded index for'}: {', '.join(qa.files()) or 'no files'}")
    print(f"Chunks: {qa.chunk_count()}.  Type a question, or 'quit' to exit.")

    history: list[tuple[str, str]] = []
    while (question := input("\nYou: ").strip()).lower() not in {"quit", "exit"}:
        if not question:
            continue
        answer = qa.ask(question, history)
        if answer.standalone_question != question:
            print(f"(searched for: {answer.standalone_question})")
        print(f"\nAssistant: {answer.text}\n\nSources:")
        for source in answer.sources:
            print(f"  {source.label()}")
        history.append((question, answer.text))


if __name__ == "__main__":
    main()
