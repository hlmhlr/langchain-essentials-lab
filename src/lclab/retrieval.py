"""Vector store, retriever and question-answering helpers (Exercise 4 and RAG).

Purpose
    Shared code for the "Embedding models", "Vector stores", "Retrievers"
    and "RetrievalQA" sections. Kept here so the scripts, the later RAG
    extension and the tests all use the same logic.

Contents
    build_vector_store()  embed chunks and store them in an in-memory Chroma
    search_documents()    the lab's Exercise 4 search function
    format_docs()         join retrieved chunks into one context string
    qa_chain()            retriever + prompt + llm: answers grounded in the documents

Note on the lab's embeddings
    The lab sets TRUNCATE_INPUT_TOKENS: 3, which cuts every chunk to its first
    3 tokens before embedding, so retrieval only compares chunk openings.
    Here each chunk is embedded in full (granite-embedding reads up to 512 tokens).
"""

import os
import uuid

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # turn off Chroma's usage reporting

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from langchain_core.runnables import RunnablePassthrough  # noqa: E402

QA_PROMPT = PromptTemplate.from_template(
    """Answer the question using only the context below.
If the context does not contain the answer, say "I don't know based on the documents."
Answer in at most three sentences.

Context:
{context}

Question: {question}

Answer:"""
)


def build_vector_store(
    chunks: list[Document], embeddings: Embeddings, name: str = "lab"
) -> Chroma:
    """Embed every chunk and keep the vectors in an in-memory Chroma collection.

    Each call gets its own collection, so two stores in one script never mix.
    """
    collection = f"{name}-{uuid.uuid4().hex[:8]}"
    store = Chroma(collection_name=collection, embedding_function=embeddings)
    if chunks:  # an empty store is allowed, e.g. for the parent document retriever
        store.add_documents(chunks)
    return store


def search_documents(retriever: BaseRetriever, query: str, top_k: int = 3) -> list[Document]:
    """Search for documents relevant to a query (lab's Exercise 4 function).

    The lab calls retriever.get_relevant_documents(), which is deprecated;
    retriever.invoke() is the current equivalent.
    """
    return retriever.invoke(query)[:top_k]


def format_docs(docs: list[Document]) -> str:
    """Join chunks into one string, labelling each with where it came from."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        label = f"[{i}] page {page + 1}" if isinstance(page, int) else f"[{i}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(parts)


def qa_chain(retriever: BaseRetriever, llm: BaseChatModel):
    """question -> retrieve chunks -> prompt with context -> model -> answer text.

    The modern (LCEL) version of the lab's RetrievalQA "stuff" chain.
    """
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
