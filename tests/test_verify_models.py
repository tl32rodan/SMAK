from types import SimpleNamespace

import scripts.verify_models as verify_models


def test_verify_models_script_runs(monkeypatch, capsys) -> None:
    class DummyEmbedder:
        def __init__(self, **_kwargs):
            return None

        def get_text_embedding(self, _text: str) -> list[float]:
            return [0.0, 1.0]

    class DummyLLM:
        def chat(self, _messages):
            return SimpleNamespace(message=SimpleNamespace(content="ok"))

    monkeypatch.setattr(verify_models, "InternalNomicEmbedding", DummyEmbedder)
    monkeypatch.setattr(verify_models, "build_internal_llm", lambda **_kwargs: DummyLLM())

    exit_code = verify_models.main(["--text", "ping", "--provider", "qwen"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Embedding vector length: 2" in output
    assert "LLM response: ok" in output
