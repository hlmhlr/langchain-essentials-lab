"""Lab sections "Embedding models" and "Vector stores".

Aim
    Turn text chunks into vectors (embeddings), store them in a vector
    database (Chroma) and find the chunks whose meaning is closest to a query.

What you'll see
    1. The size of one embedding vector and its first 5 numbers.
    2. A similarity search on the paper: the closest chunk to each query.

What to notice
    - Every chunk becomes a list of 768 numbers; similar meaning = nearby vectors.
    - Search works on meaning, not exact words: a query can find a chunk
      that uses different wording.
    - Embedding all chunks takes a few seconds; searching is instant.

Run:  uv run python scripts/09_embeddings_vectorstore.py
"""

from langchain_text_splitters import CharacterTextSplitter

from lclab.documents import load_paper
from lclab.llm import get_embeddings
from lclab.retrieval import build_vector_store
from lclab.utils import print_header

# Split the paper into chunks, as in the Documents section.
chunks = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n").split_documents(load_paper())
embeddings = get_embeddings()

# 1. Embedding model: text -> vector.
print_header("1. Embeddings")
vector = embeddings.embed_query(chunks[0].page_content)
print(f"Chunks to embed: {len(chunks)}")
print(f"Vector length: {len(vector)}")
print(f"First 5 numbers: {vector[:5]}")

# 2. Vector store: embed all chunks, then search by meaning.
print_header("2. Vector store (Chroma) similarity search")
docsearch = build_vector_store(chunks, embeddings, name="paper")

for query in ["LangChain", "What are the risks of using chatbots for mental health?"]:
    docs = docsearch.similarity_search(query)
    print(f"\nQuery: {query}")
    print(f"Closest chunk (page {docs[0].metadata['page'] + 1}): {docs[0].page_content}")
