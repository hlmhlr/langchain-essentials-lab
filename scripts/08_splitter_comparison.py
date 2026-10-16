"""Exercise 3: compare two text splitters on the PDF and the web page.

Aim
    Load both sources, split them with two different splitters, and compare
    the resulting chunks with the lab's statistics function.

What you'll see
    For each source and splitter: number of chunks, average / min / max
    chunk size, which metadata keys survived, and one example chunk.

What to notice
    - CharacterTextSplitter cuts only at one separator ("\\n"), so a long line
      can produce a chunk larger than chunk_size.
    - RecursiveCharacterTextSplitter tries paragraphs, then lines, then
      sentences, then words, so its chunks stay within chunk_size and break
      at more natural places.
    - Metadata (source, page) is copied to every chunk by both splitters.

Run:  uv run python scripts/08_splitter_comparison.py
"""

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from lclab.documents import display_document_stats, load_paper, load_web_page

# Load both sources
pdf_document = load_paper()
web_document = load_web_page()

# Two different splitters (splitter_1 is the lab's; splitter_2 is my choice)
splitter_1 = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
splitter_2 = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=60, separators=["\n\n", "\n", ". ", " ", ""]
)

for source_name, document in [("PDF", pdf_document), ("Web page", web_document)]:
    display_document_stats(splitter_1.split_documents(document), f"{source_name} / Splitter 1 (Character, 300)")
    display_document_stats(splitter_2.split_documents(document), f"{source_name} / Splitter 2 (Recursive, 600)")
