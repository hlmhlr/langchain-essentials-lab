"""Lab section "Documents": Document objects, loaders and a text splitter.

Aim
    See how LangChain represents text (a Document = page_content + metadata),
    how loaders turn a PDF and a web page into Documents, and how a text
    splitter cuts long Documents into small chunks.

What you'll see
    1. A Document made by hand, with and without metadata.
    2. The PDF loaded as one Document per page, with its page number.
    3. The web page loaded as one Document.
    4. The PDF split into ~200-character chunks, and one example chunk.

What to notice
    - Every chunk keeps the metadata of the page it came from (source, page),
      which is what later lets a RAG answer say where it found something.
    - Chunk count depends on chunk_size: smaller chunks = more chunks.

Run:  uv run python scripts/07_documents.py   (after scripts/00_fetch_data.py)
"""

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from lclab.documents import load_paper, load_web_page
from lclab.utils import print_header

# 1. Document object: text plus optional metadata.
print_header("1. Document object")
doc = Document(
    page_content="Ollama runs large language models locally. It exposes a simple HTTP API.",
    metadata={"my_document_id": 101, "my_document_source": "About Ollama"},
)
print(doc)
print(Document(page_content="Metadata is optional."))

# 2. PDF loader: one Document per page.
print_header("2. PDF loader")
document = load_paper()
print(f"Pages loaded: {len(document)}")
print(f"Page 2 metadata: {document[2].metadata}")
print(f"Page 1, first 300 characters:\n{document[1].page_content[:300]}")

# 3. Web page loader (the lab uses WebBaseLoader on the live URL; here the
#    saved copy is read with BSHTMLLoader, which gives the same kind of Document).
print_header("3. Web page loader")
web_data = load_web_page()
print(f"Characters of text: {len(web_data[0].page_content)}")
print(f"First 300 characters:\n{web_data[0].page_content.strip()[:300]}")

# 4. Text splitter: split the PDF pages into small chunks.
print_header("4. CharacterTextSplitter")
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
chunks = text_splitter.split_documents(document)
print(f"Number of chunks: {len(chunks)}")
print(f"Chunk 5: {chunks[5].page_content!r}")
print(f"Chunk 5 metadata: {chunks[5].metadata}")
