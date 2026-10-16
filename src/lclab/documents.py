"""Load the lab's two source documents and report statistics about chunks.

Purpose
    Shared code for the "Documents" section and Exercise 3. The lab loads
    the PDF and web page from the internet every time; here they are read
    from the copies saved by scripts/00_fetch_data.py, so results are the
    same on every run and work offline.

Contents
    load_paper()             the LangChain paper, one Document per PDF page
    load_web_page()          the LangChain Retrieval docs page, one Document
    display_document_stats() the lab's statistics function for Exercise 3
"""

# Note: langchain-community is being sunset (it prints a DeprecationWarning).
# These two loaders have no dedicated replacement package yet, so it is kept
# here, as in the lab; uv.lock pins a working version.
from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader
from langchain_core.documents import Document

from lclab.data import PAPER_PATH, PAPER_URL, WEB_PATH, WEB_URL


def _check_downloaded(path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/00_fetch_data.py")


def load_paper() -> list[Document]:
    """The lab's PDF, one Document per page (metadata includes the page number)."""
    _check_downloaded(PAPER_PATH)
    docs = PyPDFLoader(str(PAPER_PATH)).load()
    for doc in docs:
        doc.metadata["source"] = PAPER_URL  # show where it came from, not a local path
    return docs


def load_web_page() -> list[Document]:
    """The saved web page as one Document with the page's text."""
    _check_downloaded(WEB_PATH)
    loader = BSHTMLLoader(
        str(WEB_PATH),
        open_encoding="utf-8",
        bs_kwargs={"features": "html.parser"},  # built-in parser, no extra install
        get_text_separator="\n",
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = WEB_URL
    return docs


def display_document_stats(docs: list[Document], name: str) -> None:
    """Display statistics about a list of document chunks (from the lab)."""
    total_chunks = len(docs)
    total_chars = sum(len(doc.page_content) for doc in docs)
    avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0

    # Collect unique metadata keys across all documents
    all_metadata_keys = set()
    for doc in docs:
        all_metadata_keys.update(doc.metadata.keys())

    print(f"\n=== {name} Statistics ===")
    print(f"Total number of chunks: {total_chunks}")
    print(f"Average chunk size: {avg_chunk_size:.2f} characters")
    print(f"Metadata keys preserved: {', '.join(sorted(all_metadata_keys))}")

    if docs:
        print("\nExample chunk:")
        example_doc = docs[min(5, total_chunks - 1)]  # the 6th chunk, or the last one
        print(f"Content (first 150 chars): {example_doc.page_content[:150]}...")
        print(f"Metadata: {example_doc.metadata}")

        lengths = [len(doc.page_content) for doc in docs]
        print(f"Min chunk size: {min(lengths)} characters")
        print(f"Max chunk size: {max(lengths)} characters")
