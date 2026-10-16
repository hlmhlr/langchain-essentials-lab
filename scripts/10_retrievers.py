"""Lab sections "Retrievers" and "RetrievalQA".

Aim
    Use a vector store through the retriever interface, see how a parent
    document retriever returns more context, and answer a question from the
    paper with RetrievalQA.

What you'll see
    1. Vector store retriever: the top chunk for a query.
    2. Parent document retriever: the small matching chunk, then the larger
       "parent" chunk it belongs to, which is what gets returned.
    3. Question answering over the paper, two ways:
       the lab's RetrievalQA (legacy, from langchain-classic) and the
       modern LCEL chain from lclab.retrieval.

What to notice
    - Small chunks match a query precisely; large chunks give the model
      enough context. The parent retriever gets both.
    - RetrievalQA and the LCEL chain do the same thing ("stuff" the retrieved
      chunks into one prompt); the LCEL version shows every step explicitly.

Run:  uv run python scripts/10_retrievers.py
"""

from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import CharacterTextSplitter

from lclab.documents import load_paper
from lclab.llm import get_embeddings, get_llm
from lclab.retrieval import build_vector_store, qa_chain
from lclab.utils import print_block, print_header

document = load_paper()
embeddings = get_embeddings()
llm = get_llm()
chunks = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n").split_documents(document)
docsearch = build_vector_store(chunks, embeddings, name="paper")

# 1. Vector store-backed retriever.
print_header("1. Vector store retriever")
retriever = docsearch.as_retriever()
docs = retriever.invoke("LangChain")
print(f"Retrieved {len(docs)} chunks. Top one:\n{docs[0].page_content}")

# 2. Parent document retriever: search small chunks, return their larger parent.
print_header("2. Parent document retriever")
parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator="\n")
child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator="\n")
vectorstore = build_vector_store([], embeddings, name="split-parents")  # empty, filled below
store = InMemoryStore()

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
parent_retriever.add_documents(document)

print(f"Parent chunks stored: {len(list(store.yield_keys()))}")
sub_docs = vectorstore.similarity_search("LangChain")
print_block(f"Matching child chunk ({len(sub_docs[0].page_content)} chars)", sub_docs[0].page_content)
retrieved_docs = parent_retriever.invoke("LangChain")
print_block(
    f"Returned parent chunk ({len(retrieved_docs[0].page_content)} chars, first 500 shown)",
    retrieved_docs[0].page_content[:500],
)

# 3. Question answering over the paper.
print_header("3. Question answering")
query = "What is this paper discussing?"
print(f"Question: {query}")

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
print_block("RetrievalQA (lab, legacy)", qa.invoke(query)["result"])
print_block("LCEL qa_chain (modern)", qa_chain(docsearch.as_retriever(), llm).invoke(query))
