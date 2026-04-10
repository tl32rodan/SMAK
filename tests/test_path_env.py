"""Tests for environment variable expansion in UIDs and paths."""

from __future__ import annotations

import unittest
from pathlib import Path

from smak.utils.path_env import (
    collapse_to_env,
    contains_env_var,
    expand_env_path,
    expand_uid,
    warn_path_mismatch,
)


class TestExpandEnvPath(unittest.TestCase):
    def test_replaces_single_var(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        self.assertEqual(
            expand_env_path("$DDI_ROOT_PATH/src/a.py", env),
            "/opt/ddi/online/src/a.py",
        )

    def test_replaces_multiple_vars(self) -> None:
        env = {"HOME": "/home/user", "PROJECT": "myproj"}
        self.assertEqual(
            expand_env_path("$HOME/$PROJECT/src", env),
            "/home/user/myproj/src",
        )

    def test_passthrough_no_vars(self) -> None:
        self.assertEqual(expand_env_path("/abs/path/file.py", {}), "/abs/path/file.py")

    def test_raises_on_undefined_var(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            expand_env_path("$UNDEFINED_VAR/src", {})
        self.assertIn("UNDEFINED_VAR", str(ctx.exception))

    def test_preserves_trailing_content(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        self.assertEqual(
            expand_env_path("$DDI_ROOT_PATH/deep/nested/path.py", env),
            "/opt/ddi/online/deep/nested/path.py",
        )

    def test_empty_string(self) -> None:
        self.assertEqual(expand_env_path("", {}), "")


class TestContainsEnvVar(unittest.TestCase):
    def test_true_for_valid_var(self) -> None:
        self.assertTrue(contains_env_var("$DDI_ROOT_PATH/src"))

    def test_false_for_plain_path(self) -> None:
        self.assertFalse(contains_env_var("/abs/path"))

    def test_false_for_dollar_digit(self) -> None:
        self.assertFalse(contains_env_var("price is $5"))

    def test_true_for_var_with_digits(self) -> None:
        self.assertTrue(contains_env_var("$VAR5/path"))

    def test_false_for_empty_string(self) -> None:
        self.assertFalse(contains_env_var(""))

    def test_true_for_var_at_end(self) -> None:
        self.assertTrue(contains_env_var("/path/$SUFFIX"))


class TestCollapseToEnv(unittest.TestCase):
    def test_matches_longest_prefix(self) -> None:
        env = {
            "ROOT": "/opt",
            "DDI_ROOT_PATH": "/opt/ddi/online",
        }
        self.assertEqual(
            collapse_to_env("/opt/ddi/online/src/a.py", env),
            "$DDI_ROOT_PATH/src/a.py",
        )

    def test_returns_none_when_no_match(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        self.assertIsNone(collapse_to_env("/other/path/a.py", env))

    def test_empty_env_returns_none(self) -> None:
        self.assertIsNone(collapse_to_env("/any/path", {}))

    def test_exact_match_returns_var_only(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        self.assertEqual(
            collapse_to_env("/opt/ddi/online", env),
            "$DDI_ROOT_PATH",
        )

    def test_trailing_slash_in_value_handled(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online/"}
        result = collapse_to_env("/opt/ddi/online/src/a.py", env)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("$DDI_ROOT_PATH"))


class TestExpandUid(unittest.TestCase):
    def test_expands_path_portion(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        self.assertEqual(
            expand_uid("$DDI_ROOT_PATH/src/a.py::ClassName.method", env),
            "/opt/ddi/online/src/a.py::ClassName.method",
        )

    def test_wildcard_symbol(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        self.assertEqual(
            expand_uid("$DDI_ROOT_PATH/issues/bug.md::*", env),
            "/opt/ddi/online/issues/bug.md::*",
        )

    def test_no_env_var_passthrough(self) -> None:
        self.assertEqual(
            expand_uid("/abs/path/a.py::func", {}),
            "/abs/path/a.py::func",
        )

    def test_no_separator_expands_whole(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        self.assertEqual(
            expand_uid("$DDI_ROOT_PATH/issues/bug.md", env),
            "/opt/ddi/online/issues/bug.md",
        )

    def test_raises_on_undefined_var(self) -> None:
        with self.assertRaises(ValueError):
            expand_uid("$UNDEFINED/src/a.py::func", {})


class TestWarnPathMismatch(unittest.TestCase):
    def test_returns_warning_when_different(self) -> None:
        actual = Path("/workspace/user1/src/a.py")
        uid_path = "/opt/ddi/online/src/a.py"
        result = warn_path_mismatch(actual, uid_path, {})
        self.assertIsNotNone(result)
        self.assertIn("mismatch", result.lower())

    def test_returns_none_when_same(self) -> None:
        actual = Path("/opt/ddi/online/src/a.py")
        uid_path = "/opt/ddi/online/src/a.py"
        self.assertIsNone(warn_path_mismatch(actual, uid_path, {}))

    def test_expands_env_before_compare(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        actual = Path("/opt/ddi/online/src/a.py")
        uid_path = "$DDI_ROOT_PATH/src/a.py"
        self.assertIsNone(warn_path_mismatch(actual, uid_path, env))

    def test_warns_when_workspace_differs_from_env(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        actual = Path("/workspace/user1/src/a.py")
        uid_path = "$DDI_ROOT_PATH/src/a.py"
        result = warn_path_mismatch(actual, uid_path, env)
        self.assertIsNotNone(result)

    def test_no_env_var_no_expansion(self) -> None:
        actual = Path("/abs/path/a.py")
        uid_path = "/abs/path/a.py"
        self.assertIsNone(warn_path_mismatch(actual, uid_path, {}))


if __name__ == "__main__":
    unittest.main()
