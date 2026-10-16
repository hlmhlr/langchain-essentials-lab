"""Tests for lclab.retrieval with fake embeddings and a fake model (no Ollama)."""

from langchain_core.documents import Document

from lclab.retrieval import build_vector_store, format_docs, qa_chain, search_documents

CHUNKS = [
    Document(page_content="Retrievers return documents for a query.", metadata={"page": 0}),
    Document(page_content="Embeddings turn text into vectors.", metadata={"page": 1}),
    Document(page_content="Chunks keep context small.", metadata={"page": 2}),
]


def test_vector_store_finds_exact_text(fake_embeddings):
    # Fake embeddings give identical vectors for identical text,
    # so searching a chunk's own text must return that chunk first.
    store = build_vector_store(CHUNKS, fake_embeddings)
    top = store.similarity_search("Embeddings turn text into vectors.", k=1)[0]
    assert top.page_content == "Embeddings turn text into vectors."


def test_each_store_has_its_own_collection(fake_embeddings):
    first = build_vector_store(CHUNKS, fake_embeddings)
    second = build_vector_store(CHUNKS[:1], fake_embeddings)
    assert len(first.get()["ids"]) == 3
    assert len(second.get()["ids"]) == 1


def test_search_documents_limits_results(fake_embeddings):
    retriever = build_vector_store(CHUNKS, fake_embeddings).as_retriever(search_kwargs={"k": 3})
    assert len(search_documents(retriever, "query", top_k=2)) == 2


def test_format_docs_labels_pages():
    text = format_docs(CHUNKS[:2])
    assert "[1] page 1\nRetrievers return documents" in text
    assert "[2] page 2\nEmbeddings" in text


def test_qa_chain_puts_context_in_the_prompt(fake_embeddings, fake_llm_factory):
    retriever = build_vector_store(CHUNKS, fake_embeddings).as_retriever(search_kwargs={"k": 1})
    llm = fake_llm_factory("They return documents.")
    chain = qa_chain(retriever, llm)

    assert chain.invoke("Retrievers return documents for a query.") == "They return documents."
    # The prompt sent to the model contained the retrieved chunk and the question.
    prompt_text = (chain.first | chain.middle[0]).invoke("Retrievers return documents for a query.").to_string()
    assert "Context:\n[1] page 1\nRetrievers return documents" in prompt_text
    assert "Question: Retrievers return documents for a query." in prompt_text
