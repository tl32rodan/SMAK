import tempfile
import unittest
from pathlib import Path

from smak.utils.file_kind import is_binary_file


class TestIsBinaryFile(unittest.TestCase):
    def test_returns_true_when_null_byte_present(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(b"hello\x00world")
            path = Path(fp.name)
        try:
            self.assertTrue(is_binary_file(path))
        finally:
            path.unlink()

    def test_returns_false_for_plain_text(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(b"def foo():\n    return 1\n")
            path = Path(fp.name)
        try:
            self.assertFalse(is_binary_file(path))
        finally:
            path.unlink()

    def test_returns_false_for_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            path = Path(fp.name)
        try:
            self.assertFalse(is_binary_file(path))
        finally:
            path.unlink()

    def test_returns_false_when_path_unreadable(self) -> None:
        self.assertFalse(is_binary_file(Path("/no/such/path/exists.bin")))

    def test_only_inspects_first_sniff_window(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(b"a" * 16 + b"\x00")
            path = Path(fp.name)
        try:
            self.assertFalse(is_binary_file(path, sniff_bytes=8))
            self.assertTrue(is_binary_file(path, sniff_bytes=32))
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
