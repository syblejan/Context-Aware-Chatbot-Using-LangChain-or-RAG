# Context-Aware Chatbot Using LangChain & RAG

## Objective

This project builds a **production-style, context-aware chatbot** powered by **Retrieval-Augmented Generation (RAG)**. Rather than relying on a language model's static training knowledge, the chatbot dynamically retrieves relevant information from a user-supplied document (or a default knowledge base) and uses it as grounded context when generating answers. The system maintains full **multi-turn conversation memory**, so follow-up questions are resolved correctly even without re-stating prior context.

The chatbot is served through a **Streamlit web UI** and exposed publicly via a **Cloudflare Tunnel** — no server setup required.

---

## Architecture Overview

```
User Input
    │
    ▼
History-Aware Retriever  ←─── Chat History (session memory)
    │   (reformulates the question into a standalone query)
    │
    ▼
FAISS Vector Store  ←─── Document Chunks (embedded via Mistral Embeddings)
    │   (top-k=3 semantically similar chunks retrieved)
    │
    ▼
Stuff Documents Chain
    │   (retrieved chunks injected into QA prompt as {context})
    │
    ▼
Mistral LLM (open-mistral-7b)
    │
    ▼
Answer  ──► Displayed in Streamlit Chat UI  ──► Appended to Chat History
```

---

## Methodology / Approach

### 1. Document Ingestion & Chunking

- Users upload any `.txt` file via the sidebar, or the system falls back to a built-in ACME Corp knowledge base
- Text is split using `RecursiveCharacterTextSplitter`:
  - `chunk_size = 500`
  - `chunk_overlap = 50` (preserves cross-boundary context)

### 2. Embedding & Vector Store

- Chunks are embedded using **Mistral AI Embeddings** (`MistralAIEmbeddings`)
- Embeddings are stored in a **FAISS** in-memory vector index for fast similarity search
- At query time, the top **3 most semantically relevant chunks** are retrieved (`k=3`)

### 3. History-Aware Retrieval

A key design challenge in multi-turn RAG is **coreference resolution** — when a user asks *"What about their refund policy?"*, the retriever doesn't know what *"their"* refers to without prior context.

This is solved with a **history-aware retriever**:
- A dedicated prompt instructs the LLM to reformulate the latest user message into a fully self-contained question, given the prior chat history
- The reformulated question is then used to query the FAISS index

```
System: Given the chat history and the latest user question,
        formulate a standalone question. Do NOT answer it.
Chat History: [...]
Human: {input}
```

### 4. QA Chain & Answer Generation

The retrieved chunks are passed to a `create_stuff_documents_chain` with a professional QA prompt:

```
System: You are a polite, helpful, and professional AI assistant for ACME Corp.
        Use the provided context to answer accurately.
        Context: {context}
Chat History: [...]
Human: {input}
```

- LLM: `open-mistral-7b` via `ChatMistralAI`
- Temperature: `0.2` (low, for factual and consistent answers)

### 5. Session Memory

Chat history is maintained in `st.session_state` as a list of `HumanMessage` and `AIMessage` objects. Every turn appends both the user query and the bot response, giving the chain full conversational context on the next call.

### 6. Deployment

| Component        | Tool                          |
|------------------|-------------------------------|
| Web UI           | Streamlit (`st.chat_input`, `st.chat_message`) |
| Public Tunnel    | Cloudflare Tunnel (`trycloudflare.com`) |
| Process Management | Background thread + `pkill` for clean restarts |

The tunnel generates a public HTTPS URL directly from Colab — no port forwarding or cloud hosting needed.

---

## Tech Stack

| Component           | Tool / Library                              |
|---------------------|---------------------------------------------|
| LLM                 | `open-mistral-7b` (Mistral AI API)          |
| Embeddings          | `MistralAIEmbeddings`                       |
| Vector Store        | `FAISS` (in-memory, CPU)                    |
| RAG Framework       | `LangChain` (`langchain`, `langchain-community`, `langchain-mistralai`) |
| Text Splitting      | `RecursiveCharacterTextSplitter`            |
| Retrieval Chain     | `create_history_aware_retriever` + `create_retrieval_chain` |
| Web UI              | `Streamlit`                                 |
| Public Deployment   | Cloudflare Tunnel                           |
| Environment         | Google Colab (T4 GPU)                       |

---

## Key Results & Observations

1. **History-aware retrieval solves follow-up ambiguity.** Without query reformulation, a follow-up like *"When was it founded?"* would retrieve irrelevant chunks. The history-aware retriever converts it to *"When was ACME Corp founded?"* before hitting the vector index — dramatically improving retrieval precision.

2. **FAISS with top-k=3 keeps context tight.** Retrieving only the 3 most relevant chunks avoids flooding the LLM context with noise, while the 50-token overlap between chunks ensures no information is lost at boundaries.

3. **Temperature 0.2 balances accuracy and naturalness.** Low temperature keeps the model grounded in retrieved facts rather than hallucinating, while still producing fluent, conversational responses.

4. **`@st.cache_resource` prevents repeated RAG initialization.** The FAISS index and chain are built once and cached for the session — preventing redundant embedding API calls on every Streamlit rerender.

5. **Cloudflare Tunnel enables zero-config public access.** The chatbot is accessible via a public HTTPS URL from Colab without any cloud deployment, making it ideal for demos and rapid prototyping.

6. **Graceful error handling.** All inference calls are wrapped in `try/except`, so API key errors or network timeouts display a user-friendly warning instead of crashing the session.

---

## RAG Pipeline Configuration Summary

| Parameter              | Value                              |
|------------------------|------------------------------------|
| LLM Model              | `open-mistral-7b`                  |
| LLM Temperature        | `0.2`                              |
| Embedding Model        | `MistralAIEmbeddings`              |
| Vector Store           | FAISS (in-memory)                  |
| Retrieval Top-K        | 3                                  |
| Chunk Size             | 500 characters                     |
| Chunk Overlap          | 50 characters                      |
| Memory Type            | Session state (full message list)  |
| Retriever Type         | History-aware (query reformulation)|

---

## Repository Structure

```
.
├── context_aware_chatbot_using_langchain_&_RAG.ipynb  # Colab notebook (setup + launch)
├── app.py                                              # Streamlit app (auto-written by notebook)
└── README.md                                           # This file
```

---

## How to Run

### 1. Open in Google Colab

Upload the notebook and run cells in order.

### 2. Install Dependencies
```bash
pip install streamlit langchain langchain-core langchain-community \
            langchain-mistralai faiss-cpu requests==2.32.4
```

### 3. Configure the App

In the Streamlit sidebar:
- Enter your **Mistral API Key** (get one free at [console.mistral.ai](https://console.mistral.ai))
- Optionally upload a `.txt` knowledge base file; otherwise the default ACME Corp dataset is used

### 4. Launch the Chatbot

Run the deployment cell. A public Cloudflare URL will be printed:

```
🔗 CLICK THE LINK BELOW TO OPEN YOUR CHATBOT:
👉 https://xxxx-xxxx.trycloudflare.com 👈
```

Open the link in any browser to start chatting.

### 5. Example Interactions

```
User:  What is ACME Corp's return policy?
Bot:   Customers can return products within 30 days of purchase for a full refund.

User:  How can I contact them?
Bot:   You can reach ACME Corp's support team at support@acmecorp.com
       or by phone at 1-800-555-0199.

User:  Who founded it?
Bot:   ACME Corp was founded in 2021 by Jane Doe and John Smith.
```

> Notice: the third question uses *"it"* — the history-aware retriever correctly resolves this to *"ACME Corp"* before querying the vector store.
