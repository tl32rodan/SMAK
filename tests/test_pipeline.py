from smak.ingest.embedder import SimpleEmbedder
from smak.ingest.parsers import PythonParser
from smak.ingest.pipeline import IngestPipeline
from smak.ingest.sidecar import SidecarManager


def test_ingest_pipeline_runs_end_to_end() -> None:
    pipeline = IngestPipeline(
        parser=PythonParser(),
        embedder=SimpleEmbedder(),
        sidecar_manager=SidecarManager(),
    )

    content = "def login():\n    return True\n"
    sidecar = "symbols:\n  login:\n    relations:\n      - issue::123\n"

    result = pipeline.run(content, source="doc.txt", sidecar_payload=sidecar)

    assert [unit.metadata["symbol"] for unit in result.units] == ["login"]
    assert result.embeddings == [[28.0, 2269.0, 81.03571428571429]]
    assert result.metadata == {"symbols": {"login": {"relations": ["issue::123"]}}}
    assert result.units[0].relations == ("issue::123",)
