# Context-Aware Chatbot — LangChain + RAG

A **document-grounded conversational chatbot prototype** using Streamlit, Mistral AI, FAISS and LangChain's history-aware retriever. Upload a UTF-8 text document, ask follow-up questions and retrieve contextual answers.

> **Status:** Live Mistral RAG smoke test passed: 3/3 sample checks, plus all 8 offline helper tests and Python syntax checks. [Verified GitHub Actions run](https://github.com/syblejan/Context-Aware-Chatbot-Using-LangChain-or-RAG/actions/runs/38791545021) on October 15, 2026 (UTC). This is a limited functional check, not an independent quality benchmark or a browser UI test.

## Architecture

```text
TXT document → UTF-8 validation → 500-character chunks (50 overlap)
  → Mistral embeddings → FAISS index → top 3 retrieved chunks
                                     ↑
User question + conversation history → LLM query reformulation
                                     ↓
                 Grounded QA prompt + retrieved context → Mistral LLM
                                     ↓
                         Streamlit answer + session history
```

## Original work and improvements

The original Colab notebook is **preserved unchanged** at `notebooks/original_context_aware_rag.ipynb`. It installed LangChain packages, wrote an ephemeral `app.py` in Colab and launched Streamlit using a temporary Cloudflare Tunnel. The original repository README described this app, but **`app.py` was not committed**, so a normal clone could not run it directly.

The new standalone `app.py` addresses this and adds:
- Direct `streamlit run app.py` launch, without Colab or a public tunnel.
- Scoped Mistral credentials passed to model and embedding clients (no global environment mutation).
- UTF-8-only text uploads, **1 MB limit**, empty-document validation.
- Document fingerprints: changing the knowledge base resets the conversation to avoid mixing contexts.
- Bounded conversation context (last 12 messages) rather than unlimited prompt growth.
- Instruction to use retrieved text as **data**, not as commands, and admit unsupported answers.
- Dedicated offline unit tests for the document and history helpers.
- The default ACME Corp document is **fictional demonstration material**, not real company information.

**Limitations:** In-memory FAISS index; API cost/rate limits; no authentication or long-term storage. Retrieved passages may still contain adversarial text; prompt wording is not a full prompt-injection defense. Repeated use of a new document/key causes embedding calls. The UI displays a user-friendly error rather than publishing sensitive API exceptions.

## Run locally

```bash
git clone https://github.com/syblejan/Context-Aware-Chatbot-Using-LangChain-or-RAG.git
cd Context-Aware-Chatbot-Using-LangChain-or-RAG
python -m venv .venv
# On Windows, activate with .venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

In the Streamlit sidebar, enter a valid **Mistral API key**, and optionally upload a plain-text document. Model name `open-mistral-nemo` is an example of a Mistral model identifier; service availability, account eligibility and pricing can change. Set it to a supported model in `rag_chain.py` if needed.

### Offline tests

```bash
python -m unittest discover -s tests -v
```

Tests validate helper logic without accessing a paid API or launching Streamlit. They do **not** prove that remote embedding/inference calls work.

## Live API smoke test with GitHub Secrets

1. Open repository **Settings → Secrets and variables → Actions → New repository secret**.
2. Set the name to `MISTRAL_API_KEY` and paste the key into the **Secret** field.
3. Open **Actions → Live Mistral RAG Test → Run workflow**, choose `main`, and run.
4. Inspect the live-test step for three PASS/FAIL checks.

This manually triggered workflow uses the same `rag_chain.build_chain` as the app.
It checks the fictional 30-day return policy, a follow-up support-contact question,
and refusal of an unsupported annual-revenue question. It also checks retrieved
context for the first two questions. Checks are heuristic and can reject a valid
paraphrase; three passes are a smoke test, not a general faithfulness benchmark
or a UI/browser test.

### Verified live run

[Run 38791545021](https://github.com/syblejan/Context-Aware-Chatbot-Using-LangChain-or-RAG/actions/runs/38791545021) completed successfully on October 15, 2026 (UTC), using real Mistral embedding and chat calls on commit `a229e9284cfe92172ec20bbec54d949145b31e92`.

| Check | Result |
|---|---|
| Return-policy answer and retrieved 30-day context | PASS |
| Follow-up support-contact answer and retrieved email | PASS |
| Unsupported annual-revenue question: unknown-answer heuristic | PASS |
| Offline helper tests | 8/8 passed |
| Python compilation | Passed |

The logs record pass/fail assertions rather than raw model responses. This run validates the shared app chain on the fictional default document; it does not measure general answer accuracy, isolate the benefit of memory, or verify the Streamlit browser interface.

The workflow makes real embedding and chat API calls, which may consume account
credits. It runs only on manual dispatch, with read-only repository permissions.
The key is exposed only to credential checks and the live test via environment
variables. No raw answers, API exceptions, or credentials are intentionally logged.
Do not commit the key. Model availability and rate limits can cause a failed run.

For a local run with `MISTRAL_API_KEY` already set in your environment:
`python scripts/live_rag_smoke.py`.

## Repository structure

```text
.
├── app.py
├── rag_chain.py
├── rag_utils.py
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   └── live_rag_smoke.py
├── .github/workflows/
│   ├── offline-checks.yml
│   └── live-rag.yml
├── notebooks/
│   └── original_context_aware_rag.ipynb
├── assets/
│   └── original_demo.png
└── tests/
    └── test_rag_utils.py
```

## Original prototype versus current app

Original implementation: Mistral Chat and embeddings, FAISS, history-aware retrieval, session memory, `k=3`, Streamlit UI and a temporary Cloudflare Tunnel from Colab. The uploaded screenshot is historical evidence of a demo interface, **not** an independent quality evaluation. Claims that history-aware retrieval *dramatically improves precision*, or that every answer is grounded, require a labeled evaluation dataset and controlled measurements; neither was provided.

## Further work

Evaluation questions with ground-truth citations, retrieval metrics (Recall@k, MRR), answer faithfulness checks, source citations in the UI, persistent encrypted document storage, rate limiting, improved multi-user isolation and safer prompt-injection resistance.

## Author

**Syble Jan** — BS Computer Science, aspiring AI/ML Engineer

[GitHub profile](https://github.com/syblejan)
