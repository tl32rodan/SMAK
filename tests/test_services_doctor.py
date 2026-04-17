from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from smak.config import IndexConfig, SmakConfig
from smak.services.doctor import DoctorService


class TestDoctorService(unittest.TestCase):
    def test_shared_sidecar_suffixes_work_across_yaml_extensions(self) -> None:
        from smak.sidecar.paths import iter_sidecar_files

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / ".a.py.sidecar.yaml").write_text("symbols: []\n", encoding="utf-8")
            (root / "b.py").write_text("print('b')\n", encoding="utf-8")
            (root / ".b.py.sidecar.yml").write_text("symbols: []\n", encoding="utf-8")

            config = SmakConfig(indices=[])
            doctor = DoctorService(config=config, vector_store_loader=lambda _: object())
            issues = doctor.validate_sidecars(root)

            sidecar_files = sorted(path.name for path in iter_sidecar_files(root))

            self.assertEqual(issues, [])
            self.assertEqual(sidecar_files, [".a.py.sidecar.yaml", ".b.py.sidecar.yml"])

    def test_validate_mesh_integrity_expands_env_vars_before_lookup(self) -> None:
        """Relation UIDs containing ``$VAR`` must be expanded via ``config.env``
        before the vector-store lookup. Otherwise healthy workspaces emit false
        "does not exist in any configured vector index" warnings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            sidecar = Path(tmp_dir) / ".main.py.sidecar.yaml"
            sidecar.write_text(
                "symbols:\n"
                "  - name: main.py::hello\n"
                "    relations:\n"
                "      - $REPO_ROOT/lib.py::helper\n",
                encoding="utf-8",
            )

            expanded_uid = "/tmp/code/lib.py::helper"
            config = SmakConfig(
                indices=[IndexConfig(name="src", description="src", uri="/tmp/fake")],
                env={"REPO_ROOT": "/tmp/code"},
            )
            store = SimpleNamespace(
                get_by_id=lambda uid: {"uid": uid} if uid == expanded_uid else None
            )
            service = DoctorService(
                config=config, vector_store_loader=lambda _: store
            )

            warnings = service.validate_mesh_integrity(Path(tmp_dir))
            self.assertEqual(warnings, [])

    def test_validate_mesh_integrity_accepts_raw_uid_if_store_has_it(self) -> None:
        """A legacy store that holds the un-expanded ``$VAR/...`` UID must still
        suppress the warning. This locks in the "try raw first, then expanded"
        ordering that ``QueryService._get_payload_globally`` already uses."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("x = 1\n", encoding="utf-8")
            sidecar = Path(tmp_dir) / ".main.py.sidecar.yaml"
            raw_uid = "$REPO_ROOT/lib.py::helper"
            sidecar.write_text(
                "symbols:\n"
                "  - name: main.py::x\n"
                "    relations:\n"
                f"      - {raw_uid}\n",
                encoding="utf-8",
            )

            config = SmakConfig(
                indices=[IndexConfig(name="src", description="src", uri="/tmp/fake")],
                env={"REPO_ROOT": "/tmp/code"},
            )
            store = SimpleNamespace(
                get_by_id=lambda uid: {"uid": uid} if uid == raw_uid else None
            )
            service = DoctorService(
                config=config, vector_store_loader=lambda _: store
            )

            warnings = service.validate_mesh_integrity(Path(tmp_dir))
            self.assertEqual(warnings, [])

    def test_validate_mesh_integrity_warns_with_original_relation_text_when_neither_matches(
        self,
    ) -> None:
        """When neither the raw nor the expanded UID exists, the warning must
        contain the ORIGINAL ``$VAR/...`` text so the user can grep their YAML
        for the offending relation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("x = 1\n", encoding="utf-8")
            sidecar = Path(tmp_dir) / ".main.py.sidecar.yaml"
            raw_uid = "$REPO_ROOT/missing.py::gone"
            sidecar.write_text(
                "symbols:\n"
                "  - name: main.py::x\n"
                "    relations:\n"
                f"      - {raw_uid}\n",
                encoding="utf-8",
            )

            config = SmakConfig(
                indices=[IndexConfig(name="src", description="src", uri="/tmp/fake")],
                env={"REPO_ROOT": "/tmp/code"},
            )
            store = SimpleNamespace(get_by_id=lambda uid: None)
            service = DoctorService(
                config=config, vector_store_loader=lambda _: store
            )

            warnings = service.validate_mesh_integrity(Path(tmp_dir))
            self.assertEqual(len(warnings), 1)
            self.assertIn(raw_uid, warnings[0])

    def test_validate_mesh_integrity_handles_undefined_env_var_without_crashing(
        self,
    ) -> None:
        """An undefined env var in a relation must not crash the health check.
        ``expand_uid`` raises ``ValueError`` for unknown vars; the service must
        swallow it (as ``QueryService`` does) and fall back to the raw UID."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("x = 1\n", encoding="utf-8")
            sidecar = Path(tmp_dir) / ".main.py.sidecar.yaml"
            raw_uid = "$UNDEFINED/foo.py::bar"
            sidecar.write_text(
                "symbols:\n"
                "  - name: main.py::x\n"
                "    relations:\n"
                f"      - {raw_uid}\n",
                encoding="utf-8",
            )

            config = SmakConfig(
                indices=[IndexConfig(name="src", description="src", uri="/tmp/fake")],
                env={},
            )
            store = SimpleNamespace(get_by_id=lambda uid: None)
            service = DoctorService(
                config=config, vector_store_loader=lambda _: store
            )

            warnings = service.validate_mesh_integrity(Path(tmp_dir))
            self.assertEqual(len(warnings), 1)
            self.assertIn(raw_uid, warnings[0])

    def test_doctor_service_detects_dangling_reference_only_if_missing_in_all_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            sidecar = Path(tmp_dir) / ".main.py.sidecar.yaml"
            sidecar.write_text(
                "symbols:\n"
                "  - name: main.py::hello\n"
                "    relations:\n"
                "      - shared_uid\n"
                "      - missing_uid\n",
                encoding="utf-8",
            )

            config = SmakConfig(
                indices=[
                    IndexConfig(name="source_code", description="src", uri="/tmp/test/src"),
                    IndexConfig(name="issues", description="issues", uri="/tmp/test/issues"),
                ]
            )
            stores = {
                "source_code": SimpleNamespace(
                    get_by_id=lambda uid: {"uid": uid} if uid == "shared_uid" else None
                ),
                "issues": SimpleNamespace(get_by_id=lambda uid: None),
            }
            loader_calls: list[str] = []

            def loader(index_name: str):
                loader_calls.append(index_name)
                return stores[index_name]

            service = DoctorService(config=config, vector_store_loader=loader)
            warnings = service.validate_mesh_integrity(Path(tmp_dir))
            self.assertEqual(len(warnings), 1)
            self.assertIn("missing_uid", warnings[0])
            self.assertNotIn("shared_uid", "\n".join(warnings))
            self.assertEqual(loader_calls.count("source_code"), 1)
            self.assertEqual(loader_calls.count("issues"), 1)


if __name__ == "__main__":
    unittest.main()
