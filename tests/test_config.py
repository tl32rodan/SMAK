from smak.config import SmakConfig


def test_smak_config_defaults() -> None:
    config = SmakConfig()

    assert config.embedding_dimensions == 3
    assert config.default_source == "content"
