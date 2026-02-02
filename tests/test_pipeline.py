import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from smak.ingest.embedder import SimpleEmbedder
from smak.ingest.parsers import PythonParser
from smak.ingest.sidecar import IntegrityError, SidecarManager


def _install_fake_dependencies() -> None:
    fake_requests = ModuleType("requests")

    class FakeSession:
        def post(self, url: str, json: dict, headers: dict, timeout: float) -> SimpleNamespace:
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"data": [{"index": 0, "embedding": [0.0]}]},
            )

    fake_requests.Session = FakeSession
    fake_requests.post = lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"data": [{"index": 0, "embedding": [0.0]}]},
    )

    fake_embeddings = ModuleType("llama_index.core.embeddings")

    class FakeBaseEmbedding:
        def __init__(self, model_name: str, embed_batch_size: int) -> None:
            self.model_name = model_name
            self.embed_batch_size = embed_batch_size

        def get_text_embedding(self, text: str) -> list[float]:
            return [0.0]

        def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    fake_embeddings.BaseEmbedding = FakeBaseEmbedding

    fake_openai_like = ModuleType("llama_index.llms.openai_like")

    class FakeOpenAILike:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    fake_openai_like.OpenAILike = FakeOpenAILike

    fake_core = ModuleType("llama_index.core")
    fake_core.embeddings = fake_embeddings

    fake_llms = ModuleType("llama_index.llms")
    fake_llms.openai_like = fake_openai_like

    fake_root = ModuleType("llama_index")
    fake_root.core = fake_core
    fake_root.llms = fake_llms

    sys.modules.update(
        {
            "requests": fake_requests,
            "llama_index": fake_root,
            "llama_index.core": fake_core,
            "llama_index.core.embeddings": fake_embeddings,
            "llama_index.llms": fake_llms,
            "llama_index.llms.openai_like": fake_openai_like,
        }
    )


_install_fake_dependencies()


def _load_pipeline():
    return importlib.import_module("smak.ingest.pipeline")


class TestIngestPipeline(unittest.TestCase):
    def test_ingest_pipeline_runs_end_to_end(self) -> None:
        pipeline = _load_pipeline().IngestPipeline(
            parser=PythonParser(),
            embedder=SimpleEmbedder(),
            sidecar_manager=SidecarManager(),
        )

        content = "def login():\n    return True\n"
        sidecar = "symbols:\n  - name: login\n    relations:\n      - issue::123\n"

        result = pipeline.run(content, source="doc.txt", sidecar_payload=sidecar)

        self.assertEqual([unit.metadata["symbol"] for unit in result.units], ["login"])
        self.assertEqual(result.embeddings, [])
        self.assertEqual(
            result.metadata,
            {"symbols": [{"name": "login", "relations": ["issue::123"]}]},
        )
        self.assertEqual(result.units[0].relations, ("issue::123",))

    def test_ingest_pipeline_embeds_when_requested(self) -> None:
        pipeline = _load_pipeline().IngestPipeline(
            parser=PythonParser(),
            embedder=SimpleEmbedder(),
            sidecar_manager=SidecarManager(),
        )

        content = "def login():\n    return True\n"

        result = pipeline.run(content, source="doc.txt", compute_embeddings=True)

        self.assertEqual(result.embeddings, [[28.0, 2269.0, 81.03571428571429]])

    def test_ingest_pipeline_defaults_to_internal_nomic_embedder(self) -> None:
        class DummyEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[float(len(text))] for text in texts]

        with patch("smak.ingest.pipeline.InternalNomicEmbedding", return_value=DummyEmbedder()):
            pipeline = _load_pipeline().IngestPipeline(
                parser=PythonParser(),
                sidecar_manager=SidecarManager(),
            )

            result = pipeline.run(
                "def login():\n    return True\n",
                source="doc.txt",
                compute_embeddings=True,
            )

        self.assertEqual(result.embeddings, [[28.0]])

    def test_ingest_pipeline_skips_embedding_on_sidecar_integrity_error(self) -> None:
        embedder = SimpleNamespace(embed_documents=unittest.mock.Mock())
        embedder.embed_documents.side_effect = AssertionError("Embedding should not run")

        pipeline = _load_pipeline().IngestPipeline(
            parser=PythonParser(),
            embedder=embedder,
            sidecar_manager=SidecarManager(),
        )

        content = "def login():\n    return True\n"
        sidecar = "symbols:\n  - name: does_not_exist\n"

        with self.assertRaises(IntegrityError):
            pipeline.run(
                content,
                source="doc.txt",
                sidecar_payload=sidecar,
                compute_embeddings=True,
            )

        embedder.embed_documents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
