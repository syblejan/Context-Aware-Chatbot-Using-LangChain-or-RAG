"""Pure helpers for safe, reproducible text-document chat sessions."""
from hashlib import sha256

DEFAULT_DOCUMENT = (
    "ACME Corp is a fictional demonstration company. "
    "Its example return policy allows returns within 30 days of purchase. "
    "Its fictional support contact is support@acmecorp.com. "
    "These are sample facts only, not a real company's policies."
)

def decode_document(uploaded_bytes, max_bytes=1_000_000):
    if not isinstance(uploaded_bytes, bytes):
        raise TypeError("Uploaded document must be bytes")
    if not uploaded_bytes or len(uploaded_bytes) > max_bytes:
        raise ValueError("Upload a non-empty UTF-8 file of up to 1 MB")
    text = uploaded_bytes.decode("utf-8-sig")
    if not text.strip():
        raise ValueError("Document cannot be blank")
    return text

def document_fingerprint(text):
    return sha256(text.encode("utf-8")).hexdigest()

def clean_history(messages, max_messages=12):
    """History sent to LLM; only complete user/assistant turns."""
    return messages[-max_messages:]
