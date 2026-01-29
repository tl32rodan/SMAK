from smak.ingest.embedder import SimpleEmbedder


def test_simple_embedder_vectorizes_text() -> None:
    embedder = SimpleEmbedder()

    vectors = embedder.embed(["ab"])

    assert vectors == [[2.0, 195.0, 97.5]]


def test_simple_embedder_handles_empty_text() -> None:
    embedder = SimpleEmbedder()

    vectors = embedder.embed([""])

    assert vectors == [[0.0, 0.0, 0.0]]
