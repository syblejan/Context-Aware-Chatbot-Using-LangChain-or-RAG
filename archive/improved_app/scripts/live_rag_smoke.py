"""Opt-in live smoke test of the same RAG chain used by Streamlit."""
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not key:
        print("FAIL: Add repository Actions secret MISTRAL_API_KEY before running.")
        return 1
    # Import only after checking credentials. Offline tests never import/run this file.
    from rag_chain import build_chain
    from rag_utils import DEFAULT_DOCUMENT, clean_history
    from langchain_core.messages import HumanMessage, AIMessage

    checks = []
    try:
        chain = build_chain(DEFAULT_DOCUMENT, key)
        history = []
        cases = [
            ("document_retrieval", "What is ACME Corp's return policy?"),
            ("follow_up", "How can I contact them?"),
            ("unsupported_question", "What is the company's annual revenue?"),
        ]
        for index, (name, question) in enumerate(cases):
            # Leave room for low-rate accounts between questions.
            if index:
                time.sleep(3)
            result = chain.invoke({"input": question, "chat_history": clean_history(history)})
            answer = str(result.get("answer", "")).lower().replace("’", "'")
            context = " ".join(doc.page_content for doc in result.get("context", []))
            if index == 0:
                passed = bool(re.search(r"\b30\s+days?\b", answer)) and "30 days" in context
            elif index == 1:
                passed = "support@acmecorp.com" in answer and "support@acmecorp.com" in context
            else:
                admits_unknown = any(term in answer for term in (
                    "don't know", "do not know", "not provided", "not mentioned",
                    "does not provide", "doesn't provide", "no information",
                    "not available", "does not contain", "doesn't contain",
                    "does not include", "doesn't include", "not specified",
                ))
                passed = admits_unknown and not re.search(r"\d|\$|million|billion", answer)
            checks.append((name, bool(passed)))
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
            history.extend([HumanMessage(content=question), AIMessage(content=str(result["answer"]))])
    except Exception:
        # Do not print exception bodies, request headers, credentials, or model text.
        print("FAIL: Live API/chain call failed. Check key, credits, model availability and rate limits.")
        return 1
    success = len(checks) == 3 and all(passed for _, passed in checks)
    print("Live smoke test: " + ("PASS (3/3)" if success else "FAIL"))
    print("Three heuristic checks are not a quality benchmark or a guarantee of faithfulness.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
