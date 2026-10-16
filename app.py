import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. UI Setup ---
st.set_page_config(page_title="Enterprise RAG Chatbot", page_icon="💼", layout="centered")
st.title("CONTEXT AWARE CHATBOT")
st.markdown("Welcome! Ask me anything based on the uploaded document or our default knowledge base.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter Mistral API Key:", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Knowledge Base (.txt)", type="txt")

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Stop execution if API key is missing
if not api_key:
    st.warning("⚠️ Please enter your Mistral API Key in the sidebar to start the bot.")
    st.stop()

os.environ["MISTRAL_API_KEY"] = api_key

# --- 2. Data Loading ---
if uploaded_file is not None:
    document_text = uploaded_file.read().decode("utf-8")
    st.sidebar.success("✅ File uploaded successfully!")
else:
    document_text = """
    ACME Corp's return policy allows customers to return products within 30 days of purchase for a full refund.
    The company's customer support team can be reached at support@acmecorp.com or via phone at 1-800-555-0199.
    ACME Corp was founded in 2021 by Jane Doe and John Smith to build innovative, context-aware AI tools.
    """
    st.sidebar.info("ℹ️ Using default ACME dataset. Upload a file for custom data.")

# --- 3. RAG Pipeline Initialization ---
@st.cache_resource(show_spinner=False)
def initialize_rag(text_data, _api_key):
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([text_data])

    # Create embeddings and vector store
    embeddings = MistralAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Initialize LLM
    llm = ChatMistralAI(model="open-mistral-7b", temperature=0.2)

    # History-aware prompt
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it, just reformulate it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Balanced & Professional Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a polite, helpful, and professional AI assistant for ACME Corp. Respond to users in a warm, welcoming, and clear conversational tone, similar to an excellent human customer support agent. Use the provided context to answer accurately. Be naturally helpful, but strictly avoid overly casual slang, excessive emojis, or unprofessional jokes. Keep your answers concise and to the point.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_retriever, qa_chain)

# Load database safely
with st.spinner("🔄 Initializing AI Brain..."):
    try:
        rag_chain = initialize_rag(document_text, api_key)
    except Exception as e:
        st.error(f"Error initializing AI: {e}")
        st.stop()

# --- 4. Chat Interface & Memory ---
# Default welcome message
if "chat_history" not in st.session_state or len(st.session_state.chat_history) == 0:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am the Context Aware Chatbot. How can I help you today?")
    ]

# Display previous messages
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).markdown(msg.content)

# Handle user input
if user_input := st.chat_input("Type your question here..."):
    # Display user text
    st.chat_message("user").markdown(user_input)

    # Generate and display bot response safely
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                bot_reply = response["answer"]
                st.markdown(bot_reply)

                # Update chat memory
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=bot_reply)
                ])
            except Exception as e:
                st.error("⚠️ Sorry, I encountered a network error. Please check your API key or try again.")