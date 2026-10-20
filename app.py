"""Extension A: "Ask your documents" as a Streamlit web app.

Aim
    The same assistant as scripts/18_ask_docs.py, in a browser chat window
    (the paper used in the lab, MindGuide, also uses Streamlit for its UI).

What you'll see
    A sidebar with the indexed files and a re-index button, and a chat where
    every answer has an expandable list of sources with the matching text.

Run:  uv run --group app streamlit run app.py
"""

import streamlit as st

from lclab.doc_qa import DocQA
from lclab.llm import get_embeddings, get_llm

st.set_page_config(page_title="Ask your documents", page_icon="📄")
st.title("Ask your documents")


@st.cache_resource  # build or load the index once, not on every interaction
def load_qa(rebuild: bool = False) -> DocQA:
    return DocQA(get_llm(), get_embeddings(), rebuild=rebuild)


qa = load_qa()

with st.sidebar:
    st.subheader("Indexed files")
    for name in qa.files() or ["(none - run scripts/00_fetch_data.py)"]:
        st.write(f"- {name}")
    st.caption(f"{qa.chunk_count()} chunks")
    if st.button("Re-index"):
        load_qa.clear()
        qa = load_qa(rebuild=True)
        st.success("Index rebuilt")
    if st.button("Clear chat"):
        st.session_state.messages = []

# Each viewer's chat lives in session_state: [{"role", "content", "sources"}]
st.session_state.setdefault("messages", [])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            with st.expander(source.label()):
                st.write(source.text)

if question := st.chat_input("Ask a question about the documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Earlier (question, answer) pairs, for follow-up questions
    msgs = st.session_state.messages[:-1]
    history = [(q["content"], a["content"]) for q, a in zip(msgs[::2], msgs[1::2])]

    with st.chat_message("assistant"), st.spinner("Searching the documents..."):
        answer = qa.ask(question, history)
        if answer.standalone_question != question:
            st.caption(f"Searched for: {answer.standalone_question}")
        st.markdown(answer.text)
        for source in answer.sources:
            with st.expander(source.label()):
                st.write(source.text)

    st.session_state.messages.append({"role": "assistant", "content": answer.text, "sources": answer.sources})
