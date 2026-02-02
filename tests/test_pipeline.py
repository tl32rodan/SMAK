import unittest

from smak.ingest.embedder import SimpleEmbedder
from smak.ingest.parsers import PythonParser
from smak.ingest.pipeline import IngestPipeline
from smak.ingest.sidecar import SidecarManager


class TestIngestPipeline(unittest.TestCase):
    def test_ingest_pipeline_runs_end_to_end(self) -> None:
        pipeline = IngestPipeline(
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
        pipeline = IngestPipeline(
            parser=PythonParser(),
            embedder=SimpleEmbedder(),
            sidecar_manager=SidecarManager(),
        )

        content = "def login():\n    return True\n"

        result = pipeline.run(content, source="doc.txt", compute_embeddings=True)

        self.assertEqual(result.embeddings, [[28.0, 2269.0, 81.03571428571429]])


if __name__ == "__main__":
    unittest.main()
