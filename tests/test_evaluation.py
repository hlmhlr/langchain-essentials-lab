"""Tests for lclab.evaluation (Extension B)."""

from langchain_core.documents import Document

from lclab.evaluation import (
    EvalItem, TruncatedEmbeddings, evaluate, first_hit_rank, load_eval_items, missing_phrases, normalize,
)

DOCS = [Document(page_content="Streamlit is an open-\nsource   Python library."),
        Document(page_content="Chains break down complex tasks.")]


def test_normalize_rejoins_hyphenated_lines_and_spaces():
    assert normalize(DOCS[0].page_content) == "streamlit is an opensource python library."


def test_first_hit_rank():
    assert first_hit_rank(DOCS, "Break down COMPLEX tasks") == 2
    assert first_hit_rank(DOCS, "not there") is None


def test_missing_phrases():
    items = [EvalItem("q1", "complex tasks"), EvalItem("q2", "quantum")]
    assert missing_phrases(items, DOCS) == [items[1]]


def test_evaluate_scores():
    items = [EvalItem("a", "first"), EvalItem("b", "third"), EvalItem("c", "nowhere")]
    ranked = [Document(page_content=t) for t in ["first", "second", "third", "x", "y"]]
    scores = evaluate(lambda q: ranked, items, ks=(1, 3))
    assert scores["hit@1"] == 1 / 3
    assert scores["hit@3"] == 2 / 3
    assert abs(scores["MRR"] - (1 + 1 / 3) / 3) < 1e-9


def test_truncated_embeddings_cut_every_text(fake_embeddings):
    truncated = TruncatedEmbeddings(fake_embeddings, n_words=3)
    assert truncated.embed_query("one two three four") == fake_embeddings.embed_query("one two three")
    assert truncated.embed_documents(["a b c d e"]) == fake_embeddings.embed_documents(["a b c"])


def test_eval_file_is_valid():
    items = load_eval_items()
    assert len(items) == 10
    assert all(item.question.endswith("?") and item.phrase for item in items)
