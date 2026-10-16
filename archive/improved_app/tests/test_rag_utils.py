import unittest
from rag_utils import DEFAULT_DOCUMENT, decode_document, document_fingerprint, clean_history

class RAGHelperTests(unittest.TestCase):
    def test_utf8_decode(self):
        self.assertEqual(decode_document(b"test knowledge"), "test knowledge")
    def test_bom_decode(self):
        self.assertEqual(decode_document(b"\xef\xbb\xbfhello"), "hello")
    def test_empty_document_rejected(self):
        with self.assertRaises(ValueError):
            decode_document(b" ")
    def test_invalid_encoding_rejected(self):
        with self.assertRaises(UnicodeDecodeError):
            decode_document(b"\xff")
    def test_limit(self):
        with self.assertRaises(ValueError):
            decode_document(b"x" * 1001, max_bytes=1000)
    def test_fingerprint_changes(self):
        self.assertNotEqual(document_fingerprint("one"), document_fingerprint("two"))
    def test_history_bounded(self):
        self.assertEqual(clean_history(list(range(20))), list(range(8,20)))
    def test_fictional_default(self):
        self.assertIn("fictional", DEFAULT_DOCUMENT)

if __name__ == "__main__":
    unittest.main()
