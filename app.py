"""Streamlit demo of history-aware retrieval augmented generation."""
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_chain import build_chain
from rag_utils import DEFAULT_DOCUMENT, decode_document, document_fingerprint, clean_history

st.set_page_config(page_title="Context-Aware RAG Chatbot", page_icon="💬")
st.title("Context-Aware RAG Chatbot")
st.caption("Educational document Q&A with Mistral, FAISS and conversational retrieval.")

key = st.sidebar.text_input("Mistral API key", type="password", value=os.environ.get("MISTRAL_API_KEY", ""))
file = st.sidebar.file_uploader("Knowledge base (.txt, UTF-8, max 1 MB)", type=["txt"])
if st.sidebar.button("Clear conversation"):
    st.session_state.history = []
    st.rerun()

try:
    document = decode_document(file.getvalue()) if file is not None else DEFAULT_DOCUMENT
except (UnicodeDecodeError, TypeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
fingerprint = document_fingerprint(document)
if st.session_state.get("document_fingerprint") != fingerprint:
    st.session_state.document_fingerprint = fingerprint
    st.session_state.history = []
    st.sidebar.info("Knowledge base changed: chat history cleared.")
if not key:
    st.info("Enter your Mistral API key in the sidebar to start.")
    st.stop()

@st.cache_resource(show_spinner=False, max_entries=3)
def make_chain(text, document_id, api_key):
    return build_chain(text, api_key)

try:
    chain = make_chain(document, fingerprint, key)
except Exception:
    st.error("Could not initialize retrieval. Check credentials, model availability and network connectivity.")
    st.stop()
history = st.session_state.setdefault("history", [])
for message in history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.write(message.content)
question = st.chat_input("Ask a question about the document")
if question:
    st.chat_message("user").write(question)
    try:
        response = chain.invoke({"input": question, "chat_history": clean_history(history)})
        answer = str(response["answer"])
        st.chat_message("assistant").write(answer)
        history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    except Exception:
        st.error("Unable to generate answer. Verify API access and retry.")
