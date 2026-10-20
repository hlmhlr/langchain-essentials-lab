"""Extension B: which chunking setting retrieves the right text most often?

Aim
    Turn Exercise 3's "compare the splitters" into a measurement, and show
    what the lab's 3-token embedding truncation does to retrieval.

What you'll see
    1. A check that every test phrase really occurs in the paper (any that
       don't are listed so you can fix data/eval_questions.json).
    2. A table: one row per setting, with hit@1, hit@3, hit@5 and MRR over
       the test questions. Saved to outputs/retrieval_eval.md.

What to notice
    - Truncating every chunk and query to 3 words should score far lower:
      the embeddings only see the first words of each chunk.
    - Larger chunks are more likely to contain the phrase, but each one
      also mixes more topics; compare hit@1 with hit@5.
    - With 10 questions, one question = 10 percentage points, so treat small
      differences as noise.

Run:  uv run python scripts/19_retrieval_eval.py
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from lclab.data import ROOT
from lclab.documents import load_paper
from lclab.evaluation import TruncatedEmbeddings, evaluate, in_memory_retriever, load_eval_items, missing_phrases
from lclab.llm import get_embeddings
from lclab.utils import print_header

OUTPUT = ROOT / "outputs" / "retrieval_eval.md"

# (name, splitter, truncate to 3 words?)
CONFIGS = [
    ("Character 200/20, truncated to 3 words (lab)", CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n"), True),
    ("Character 200/20", CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n"), False),
    ("Character 300/30 (Ex. 3 splitter 1)", CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n"), False),
    ("Recursive 600/60 (Ex. 3 splitter 2)", RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60), False),
    ("Recursive 1000/100", RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100), False),
]


def main() -> None:
    document = load_paper()
    items = load_eval_items()
    embeddings = get_embeddings()

    print_header("1. Checking the test data")
    missing = missing_phrases(items, document)
    for item in missing:
        print(f"[SKIP] phrase not found in the paper: {item.phrase!r}  ({item.question})")
    items = [item for item in items if item not in missing]
    print(f"{len(items)} questions usable.")

    print_header("2. Retrieval scores")
    rows = ["| Setting | Chunks | hit@1 | hit@3 | hit@5 | MRR |", "|---|---|---|---|---|---|"]
    for name, splitter, truncate in CONFIGS:
        chunks = splitter.split_documents(document)
        emb = TruncatedEmbeddings(embeddings, n_words=3) if truncate else embeddings
        retriever = in_memory_retriever(chunks, emb, k=5)
        s = evaluate(retriever.invoke, items)
        rows.append(f"| {name} | {len(chunks)} | {s['hit@1']:.0%} | {s['hit@3']:.0%} | "
                    f"{s['hit@5']:.0%} | {s['MRR']:.2f} |")
        print(rows[-1])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(f"# Retrieval evaluation ({len(items)} questions)\n\n" + "\n".join(rows) + "\n",
                      encoding="utf-8")
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
