from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from smak.services.doctor import DoctorService


class TestDoctorService(unittest.TestCase):
    def test_shared_sidecar_suffixes_work_across_yaml_extensions(self) -> None:
        from smak.sidecar.paths import iter_sidecar_files

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / "a.py.sidecar.yaml").write_text("symbols: []\n", encoding="utf-8")
            (root / "b.py").write_text("print('b')\n", encoding="utf-8")
            (root / "b.py.sidecar.yml").write_text("symbols: []\n", encoding="utf-8")

            doctor = DoctorService()
            issues = doctor.validate_sidecars(root)

            sidecar_files = sorted(path.name for path in iter_sidecar_files(root))

            self.assertEqual(issues, [])
            self.assertEqual(sidecar_files, ["a.py.sidecar.yaml", "b.py.sidecar.yml"])

    def test_doctor_service_detects_dangling_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            sidecar = Path(tmp_dir) / "main.py.sidecar.yaml"
            sidecar.write_text(
                "symbols:\n  - name: main.py::hello\n    relations:\n      - missing_uid\n",
                encoding="utf-8",
            )
            service = DoctorService(vector_store=SimpleNamespace(get_by_id=lambda uid: None))
            warnings = service.validate_mesh_integrity(Path(tmp_dir))
            self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
