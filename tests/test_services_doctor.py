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
