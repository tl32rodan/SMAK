import unittest

from smak.core.domain import KnowledgeUnit
from smak.ingest.sidecar import IntegrityError, SidecarManager


class TestSidecarManager(unittest.TestCase):
    def test_sidecar_loader_none_returns_empty(self) -> None:
        loader = SidecarManager()

        self.assertEqual(loader.load(None), {})

    def test_sidecar_loader_parses_yaml(self) -> None:
        loader = SidecarManager()

        self.assertEqual(loader.load("key: value"), {"key": "value"})

    def test_sidecar_loader_wraps_non_dict_yaml(self) -> None:
        loader = SidecarManager()

        self.assertEqual(loader.load("- a\n- b\n"), {"value": ["a", "b"]})

    def test_sidecar_validate_missing_symbol(self) -> None:
        loader = SidecarManager()
        metadata = {
            "symbols": [
                {"name": "missing", "relations": ["issue:1"], "intent": "missing-intent"}
            ]
        }

        with self.assertRaises(IntegrityError):
            loader.validate(["present"], metadata)

    def test_sidecar_validate_rejects_invalid_schema(self) -> None:
        loader = SidecarManager()
        metadata = {"symbols": [{"intent": "missing-name"}]}

        with self.assertRaises(IntegrityError):
            loader.validate(["present"], metadata)

    def test_sidecar_apply_enriches_units(self) -> None:
        loader = SidecarManager()
        units = [
            KnowledgeUnit(
                uid="main.py::login",
                content="def login(): pass",
                source_type="source_code",
                metadata={"symbol": "login", "source": "main.py"},
            )
        ]
        metadata = {
            "symbols": [
                {
                    "name": "login",
                    "relations": ["issue:404"],
                    "intent": "auth",
                    "owner": "team",
                }
            ]
        }

        enriched = loader.apply(units, metadata)

        self.assertEqual(enriched[0].relations, ("issue:404",))
        self.assertEqual(enriched[0].metadata["owner"], "team")
        self.assertEqual(enriched[0].metadata["file_name"], "main.py")
        self.assertEqual(enriched[0].metadata["symbol_name"], "login")
        self.assertEqual(enriched[0].metadata["intent"], "auth")
        self.assertEqual(enriched[0].metadata["mesh_relations"], ["issue:404"])

    def test_sidecar_apply_uses_existing_relations_when_missing(self) -> None:
        loader = SidecarManager()
        units = [
            KnowledgeUnit(
                uid="main.py::logout",
                content="def logout(): pass",
                source_type="source_code",
                relations=("issue:200",),
                metadata={"symbol": "logout", "source": "main.py"},
            )
        ]
        metadata = {"symbols": []}

        enriched = loader.apply(units, metadata)

        self.assertEqual(enriched[0].relations, ("issue:200",))
        self.assertEqual(enriched[0].metadata["mesh_relations"], ["issue:200"])

    def test_sidecar_apply_inherits_class_relations_to_methods(self) -> None:
        loader = SidecarManager()
        units = [
            KnowledgeUnit(
                uid="src/auth.py::Auth",
                content="class Auth: ...",
                source_type="source_code",
                metadata={"symbol": "Auth", "source": "src/auth.py", "symbol_type": "class"},
            ),
            KnowledgeUnit(
                uid="src/auth.py::Auth.login",
                content="def login(self): ...",
                source_type="source_code",
                metadata={
                    "symbol": "Auth.login",
                    "source": "src/auth.py",
                    "symbol_type": "method",
                    "parent_class": "Auth",
                },
            ),
        ]
        metadata = {"symbols": [{"name": "Auth", "relations": ["issue:101"]}]}

        enriched = loader.apply(units, metadata)

        self.assertEqual(enriched[1].metadata["mesh_relations"], ["issue:101"])
        self.assertEqual(enriched[1].relations, ("issue:101",))


if __name__ == "__main__":
    unittest.main()
