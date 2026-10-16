"""Exercise 4: a simple retrieval system over the LangChain Retrieval docs page.

Aim
    Build the full pipeline: load -> split -> embed -> vector store ->
    retriever, test it with three questions, then answer each question from
    the retrieved chunks.

What you'll see
    For each question: the top 3 chunks found (first 150 characters each),
    then the model's answer based only on those chunks.

What to notice
    - The lab's solution stops after retrieval; the answer step here is what
      turns a search engine into a question-answering (RAG) system.
    - If the chunks don't contain the answer, the prompt tells the model to
      say so instead of inventing one.
    - Answer quality depends on retrieval: a wrong chunk means a weak answer.

Run:  uv run python scripts/11_retrieval_qa.py
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from lclab.documents import load_web_page
from lclab.llm import get_embeddings, get_llm
from lclab.retrieval import build_vector_store, qa_chain, search_documents
from lclab.utils import print_block, print_header

# 1. Load a document about AI
documents = load_web_page()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3-4. Embed the chunks and create a vector store
vector_store = build_vector_store(chunks, get_embeddings(), name="web")

# 5. Create a retriever (top 3 chunks per query)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6. Question answering on top of the retriever
answer_chain = qa_chain(retriever, get_llm())

# 7. Test with my own questions
test_queries = [
    "What is retrieval-augmented generation?",
    "What is the difference between 2-step RAG and agentic RAG?",
    "Why do we split documents into chunks?",
]

print(f"Chunks in the vector store: {len(chunks)}")
for query in test_queries:
    print_header(f"Query: {query}")
    results = search_documents(retriever, query)
    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results, start=1):
        snippet = " ".join(doc.page_content.split())[:150]  # collapse line breaks
        print(f"  Result {i}: {snippet}...")
    print_block("Answer", answer_chain.invoke(query))
