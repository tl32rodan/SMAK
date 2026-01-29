from smak.ingest.embedder import SimpleEmbedder
from smak.ingest.parsers import SimpleLineParser
from smak.ingest.pipeline import IngestPipeline
from smak.ingest.sidecar import SidecarLoader


def test_ingest_pipeline_runs_end_to_end() -> None:
    pipeline = IngestPipeline(
        parser=SimpleLineParser(),
        embedder=SimpleEmbedder(),
        sidecar_loader=SidecarLoader(),
    )

    result = pipeline.run("line", source="doc.txt", sidecar_payload='{"tag": "x"}')

    assert [unit.content for unit in result.units] == ["line"]
    assert result.embeddings == [[4.0, 424.0, 106.0]]
    assert result.metadata == {"tag": "x"}
