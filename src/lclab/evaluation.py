"""Extension B: measure retrieval quality instead of eyeballing it.

Purpose
    Exercise 3 compares splitters by chunk statistics; this checks which
    setting actually finds the right text. Each test question has a short
    phrase that appears in the paper next to its answer. A question counts
    as a hit@k if one of the top k retrieved chunks contains that phrase.

Metrics
    hit@k  share of questions with the answer phrase in the top k chunks
    MRR    mean reciprocal rank: 1/rank of the first chunk with the phrase
           (1.0 = always first, 0.5 = on average second, 0 = never found)

Contents
    load_eval_items()      read data/eval_questions.json
    normalize()            lowercase, rejoin hyphenated line breaks, collapse spaces
    missing_phrases()      questions whose phrase isn't in the document at all
    first_hit_rank()       position of the first chunk containing the phrase
    evaluate()             hit@k and MRR for one retriever
    TruncatedEmbeddings    reproduces the lab's TRUNCATE_INPUT_TOKENS: 3 setting
    in_memory_retriever()  quick vector search without Chroma
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from lclab.data import ROOT

EVAL_FILE = ROOT / "data" / "eval_questions.json"


@dataclass
class EvalItem:
    question: str
    phrase: str


def load_eval_items(path: Path = EVAL_FILE) -> list[EvalItem]:
    return [EvalItem(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def normalize(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)  # "retrie-\nval" -> "retrieval"
    return re.sub(r"\s+", " ", text).strip().lower()


def missing_phrases(items: list[EvalItem], docs: list[Document]) -> list[EvalItem]:
    """Questions whose phrase never appears in the documents (bad test data)."""
    full_text = normalize("\n".join(doc.page_content for doc in docs))
    return [item for item in items if normalize(item.phrase) not in full_text]


def first_hit_rank(docs: list[Document], phrase: str) -> int | None:
    """1-based position of the first chunk containing `phrase`, or None."""
    target = normalize(phrase)
    for rank, doc in enumerate(docs, start=1):
        if target in normalize(doc.page_content):
            return rank
    return None


def evaluate(retrieve: Callable[[str], list[Document]], items: list[EvalItem],
             ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    """`retrieve(question)` must return at least max(ks) chunks, best first."""
    ranks = [first_hit_rank(retrieve(item.question)[:max(ks)], item.phrase) for item in items]
    scores = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / len(items) for k in ks}
    scores["MRR"] = sum(1 / r for r in ranks if r) / len(items)
    return scores


class TruncatedEmbeddings(Embeddings):
    """Embed only the first `n_words` words of every text, like the lab.

    The lab's TRUNCATE_INPUT_TOKENS: 3 cuts every input (chunks and queries)
    to 3 tokens before embedding. Words are used here as an approximation.
    """

    def __init__(self, base: Embeddings, n_words: int = 3):
        self.base, self.n_words = base, n_words

    def _cut(self, text: str) -> str:
        return " ".join(text.split()[: self.n_words])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self._cut(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self._cut(text))


def in_memory_retriever(chunks: list[Document], embeddings: Embeddings, k: int = 5):
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return store.as_retriever(search_kwargs={"k": k})
