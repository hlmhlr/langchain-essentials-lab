"""Extension A: "Ask your documents", a RAG assistant over a folder of files.

Purpose
    Combines the lab's building blocks into one usable tool:
    loaders -> splitter -> embeddings -> persistent Chroma -> retriever
    -> prompt -> model, plus chat history for follow-up questions.

What it adds on top of Exercise 4
    - Any folder of .pdf / .html / .txt / .md files, not one fixed document.
    - Persistent index: chunks are embedded once and saved in chroma_db/.
      The index is rebuilt only when the files change.
    - Citations: every answer lists its sources (file and page) as [1], [2], ...
    - "I don't know" when the documents don't contain the answer.
    - Follow-up questions: "what about its memory?" is first rewritten into a
      standalone question using the chat history, then searched.

Contents
    load_folder()   read every supported file in a folder into Documents
    DocQA           build or reuse the index, and answer questions with sources
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # turn off Chroma's usage reporting

from langchain_chroma import Chroma  # noqa: E402
from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from lclab.data import DOWNLOAD_DIR, ROOT  # noqa: E402

DEFAULT_FOLDER = DOWNLOAD_DIR
DEFAULT_DB = ROOT / "chroma_db"
SUPPORTED = {".pdf", ".html", ".htm", ".txt", ".md"}
NO_ANSWER = "I don't know based on the documents."

ANSWER_PROMPT = PromptTemplate.from_template(
    """Answer the question using only the numbered sources below.
Cite the sources you used like [1] or [2].
If the sources do not contain the answer, reply exactly: "{no_answer}"

Sources:
{context}

Question: {question}

Answer:"""
)

REWRITE_PROMPT = PromptTemplate.from_template(
    """Rewrite the follow-up question so it can be understood without the conversation.
Return only the rewritten question.

Conversation:
{history}

Follow-up question: {question}

Standalone question:"""
)


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    elif suffix in {".html", ".htm"}:
        docs = BSHTMLLoader(str(path), open_encoding="utf-8", bs_kwargs={"features": "html.parser"},
                            get_text_separator="\n").load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()
    for doc in docs:
        doc.metadata["source"] = path.name  # short name for citations
    return docs


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)


def load_folder(folder: Path) -> list[Document]:
    """Every supported file in `folder` as Documents (one per PDF page)."""
    return [doc for path in _files(folder) for doc in _load_file(path)]


def _fingerprint(folder: Path, chunk_size: int, chunk_overlap: int) -> str:
    """Changes whenever a file is added, removed or edited, or the chunking changes."""
    parts = [f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in _files(folder)]
    parts.append(f"chunks:{chunk_size}:{chunk_overlap}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


@dataclass
class Source:
    number: int
    file: str
    page: int | None  # 1-based page for PDFs, None for other files
    text: str

    def label(self) -> str:
        return f"[{self.number}] {self.file}" + (f", page {self.page}" if self.page else "")


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    standalone_question: str = ""


class DocQA:
    """Question answering over a folder, with a persistent vector index."""

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings, folder: Path = DEFAULT_FOLDER,
                 persist_dir: Path = DEFAULT_DB, k: int = 4, rebuild: bool = False,
                 chunk_size: int = 800, chunk_overlap: int = 100):
        self.llm, self.folder, self.k = llm, Path(folder), k
        self.store = Chroma(collection_name="docs", embedding_function=embeddings,
                            persist_directory=str(persist_dir))

        stamp_file = Path(persist_dir) / "index.json"
        fingerprint = _fingerprint(self.folder, chunk_size, chunk_overlap)
        old = json.loads(stamp_file.read_text()).get("fingerprint") if stamp_file.exists() else None

        self.rebuilt = rebuild or old != fingerprint
        if self.rebuilt:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(load_folder(self.folder))
            self.store.reset_collection()
            if chunks:
                self.store.add_documents(chunks)
            stamp_file.write_text(json.dumps({"fingerprint": fingerprint}))

        self.retriever = self.store.as_retriever(search_kwargs={"k": k})

    def files(self) -> list[str]:
        return [p.name for p in _files(self.folder)]

    def chunk_count(self) -> int:
        return len(self.store.get(include=[])["ids"])

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> Answer:
        """Answer `question`. `history` = earlier (question, answer) pairs, newest last."""
        standalone = question
        if history:
            conversation = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])
            standalone = (REWRITE_PROMPT | self.llm | StrOutputParser()).invoke(
                {"history": conversation, "question": question}).strip()

        docs = self.retriever.invoke(standalone)
        if not docs:
            return Answer(NO_ANSWER, [], standalone)

        sources = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            sources.append(Source(i, doc.metadata.get("source", "?"),
                                  page + 1 if isinstance(page, int) else None, doc.page_content))
        context = "\n\n".join(f"{s.label()}\n{s.text}" for s in sources)

        text = (ANSWER_PROMPT | self.llm | StrOutputParser()).invoke(
            {"context": context, "question": standalone, "no_answer": NO_ANSWER})
        return Answer(text.strip(), sources, standalone)
