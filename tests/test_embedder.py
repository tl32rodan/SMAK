import unittest

from smak.ingest.embedder import SimpleEmbedder


class TestSimpleEmbedder(unittest.TestCase):
    def test_simple_embedder_vectorizes_text(self) -> None:
        embedder = SimpleEmbedder()

        vectors = embedder.embed(["ab"])

        self.assertEqual(vectors, [[2.0, 195.0, 97.5]])

    def test_simple_embedder_handles_empty_text(self) -> None:
        embedder = SimpleEmbedder()

        vectors = embedder.embed([""])

        self.assertEqual(vectors, [[0.0, 0.0, 0.0]])

    def test_simple_embedder_embeds_documents(self) -> None:
        embedder = SimpleEmbedder()

        vectors = embedder.embed_documents(["ab"])

        self.assertEqual(vectors, [[2.0, 195.0, 97.5]])


if __name__ == "__main__":
    unittest.main()
