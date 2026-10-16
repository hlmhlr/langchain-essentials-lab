"""Tests for lclab.documents: loading the saved web page and chunk statistics."""

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lclab import documents


def test_load_web_page_reads_saved_html(tmp_path, monkeypatch):
    page = tmp_path / "page.html"
    page.write_text(
        "<html><head><title>Retrieval</title><script>var x = 1;</script></head>"
        "<body><h1>Retrieval</h1><p>Retrievers return documents.</p></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr(documents, "WEB_PATH", page)

    docs = documents.load_web_page()
    assert len(docs) == 1
    assert "Retrievers return documents." in docs[0].page_content
    assert "var x" not in docs[0].page_content  # script code is not text
    assert docs[0].metadata["source"] == documents.WEB_URL


def test_missing_download_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "PAPER_PATH", tmp_path / "missing.pdf")
    with pytest.raises(FileNotFoundError, match="00_fetch_data"):
        documents.load_paper()


def test_splitting_keeps_metadata_and_size():
    doc = Document(page_content="word " * 300, metadata={"source": "x", "page": 3})
    chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10).split_documents([doc])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)
    assert all(c.metadata == {"source": "x", "page": 3} for c in chunks)


def test_display_document_stats(capsys):
    docs = [Document(page_content="a" * 10, metadata={"page": 1}),
            Document(page_content="b" * 30, metadata={"source": "y"})]
    documents.display_document_stats(docs, "Test")
    out = capsys.readouterr().out
    assert "Total number of chunks: 2" in out
    assert "Average chunk size: 20.00" in out
    assert "Metadata keys preserved: page, source" in out
    assert "Min chunk size: 10" in out and "Max chunk size: 30" in out
