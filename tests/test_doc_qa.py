"""Tests for lclab.doc_qa (Extension A) with fake models and a temporary folder."""

import pytest

from lclab.doc_qa import NO_ANSWER, DocQA, load_folder


class CountingEmbeddings:
    """Wraps fake embeddings and counts how many texts were embedded."""

    def __init__(self, base):
        self.base, self.embedded = base, 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return self.base.embed_documents(texts)

    def embed_query(self, text):
        return self.base.embed_query(text)


@pytest.fixture
def docs_folder(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "streamlit.md").write_text("Streamlit is an open-source Python library for web apps.")
    (folder / "chains.txt").write_text("Chains break down complex tasks into smaller steps.")
    (folder / "ignored.csv").write_text("not,a,supported,file")
    return folder


def make_qa(folder, tmp_path, embeddings, llm, **kwargs):
    return DocQA(llm, embeddings, folder=folder, persist_dir=tmp_path / "db", k=2, **kwargs)


def test_load_folder_reads_supported_files_only(docs_folder):
    docs = load_folder(docs_folder)
    assert sorted(d.metadata["source"] for d in docs) == ["chains.txt", "streamlit.md"]


def test_answer_has_numbered_sources(docs_folder, tmp_path, fake_embeddings, fake_llm_factory):
    qa = make_qa(docs_folder, tmp_path, fake_embeddings, fake_llm_factory("It is a Python library [1]."))
    answer = qa.ask("Streamlit is an open-source Python library for web apps.")

    assert answer.text == "It is a Python library [1]."
    assert [s.number for s in answer.sources] == [1, 2]
    assert answer.sources[0].label() == "[1] streamlit.md"  # exact text -> found first
    assert answer.standalone_question == answer.sources[0].text


def test_index_is_reused_until_files_change(docs_folder, tmp_path, fake_embeddings, fake_llm_factory):
    counting = CountingEmbeddings(fake_embeddings)
    first = make_qa(docs_folder, tmp_path, counting, fake_llm_factory())
    assert first.rebuilt and counting.embedded == 2 and first.chunk_count() == 2

    second = make_qa(docs_folder, tmp_path, counting, fake_llm_factory())
    assert not second.rebuilt and counting.embedded == 2  # nothing embedded again

    (docs_folder / "new.txt").write_text("A new document.")
    third = make_qa(docs_folder, tmp_path, counting, fake_llm_factory())
    assert third.rebuilt and third.chunk_count() == 3


def test_follow_up_is_rewritten_before_searching(docs_folder, tmp_path, fake_embeddings, fake_llm_factory):
    llm = fake_llm_factory("What do chains do?", "They split tasks [1].")
    qa = make_qa(docs_folder, tmp_path, fake_embeddings, llm)

    answer = qa.ask("And what do they do?", history=[("Tell me about chains", "Chains are...")])
    assert answer.standalone_question == "What do chains do?"
    assert answer.text == "They split tasks [1]."


def test_empty_folder_answers_i_dont_know(tmp_path, fake_embeddings, fake_llm_factory):
    empty = tmp_path / "empty"
    empty.mkdir()
    qa = make_qa(empty, tmp_path, fake_embeddings, fake_llm_factory())
    assert qa.ask("Anything?").text == NO_ANSWER
