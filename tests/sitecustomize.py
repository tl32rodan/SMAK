"""Test-environment dependency shims for offline execution."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any


def _install_requests() -> None:
    if "requests" in sys.modules:
        return
    mod = ModuleType("requests")

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"data": [{"index": 0, "embedding": [0.0]}]}

    class Session:
        def post(self, *_args: Any, **_kwargs: Any) -> _Resp:
            return _Resp()

    mod.Session = Session
    sys.modules["requests"] = mod


def _install_yaml() -> None:
    if "yaml" in sys.modules:
        return
    mod = ModuleType("yaml")

    def _parse_scalar(value: str) -> Any:
        raw = value.strip()
        if raw == "":
            return ""
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if inner == "":
                return []
            return [_parse_scalar(item) for item in inner.split(",")]
        if raw.isdigit():
            return int(raw)
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            return raw[1:-1]
        return raw

    def _split_kv(text: str) -> tuple[str, str]:
        key, value = text.split(":", 1)
        return key.strip(), value.strip()

    def _parse_block(lines: list[str], idx: int, indent: int) -> tuple[Any, int]:
        if idx >= len(lines):
            return {}, idx
        stripped = lines[idx].lstrip(" ")
        cur_indent = len(lines[idx]) - len(stripped)
        if cur_indent < indent:
            return {}, idx
        if stripped.startswith("- "):
            return _parse_list(lines, idx, indent)
        return _parse_dict(lines, idx, indent)

    def _parse_list(lines: list[str], idx: int, indent: int) -> tuple[list[Any], int]:
        out: list[Any] = []
        i = idx
        while i < len(lines):
            raw = lines[i]
            stripped = raw.lstrip(" ")
            cur_indent = len(raw) - len(stripped)
            if cur_indent < indent or not stripped.startswith("- "):
                break
            content = stripped[2:].strip()
            if content == "":
                nested, i = _parse_block(lines, i + 1, indent + 2)
                out.append(nested)
                continue
            if ": " in content or content.endswith(":"):
                key, value = _split_kv(content)
                obj: dict[str, Any] = {key: _parse_scalar(value)} if value else {key: None}
                i += 1
                while i < len(lines):
                    nraw = lines[i]
                    nstrip = nraw.lstrip(" ")
                    nindent = len(nraw) - len(nstrip)
                    if nindent <= indent or nstrip.startswith("- "):
                        break
                    nk, nv = _split_kv(nstrip)
                    if nv:
                        obj[nk] = _parse_scalar(nv)
                        i += 1
                    else:
                        nested, i = _parse_block(lines, i + 1, nindent + 2)
                        obj[nk] = nested
                out.append(obj)
                continue
            out.append(_parse_scalar(content))
            i += 1
        return out, i

    def _parse_dict(lines: list[str], idx: int, indent: int) -> tuple[dict[str, Any], int]:
        out: dict[str, Any] = {}
        i = idx
        while i < len(lines):
            raw = lines[i]
            stripped = raw.lstrip(" ")
            cur_indent = len(raw) - len(stripped)
            if cur_indent < indent:
                break
            if cur_indent > indent or ":" not in stripped:
                i += 1
                continue
            key, value = _split_kv(stripped)
            if value:
                out[key] = _parse_scalar(value)
                i += 1
                continue
            nested, next_i = _parse_block(lines, i + 1, indent + 2)
            out[key] = nested
            i = next_i
        return out, i

    def safe_load(text: str) -> Any:
        lines = [
            ln.rstrip()
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not lines:
            return {}
        value, _ = _parse_block(lines, 0, 0)
        return value

    def _dump_scalar(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        txt = str(value)
        return f'"{txt}"' if txt == "" else txt

    def _dump(data: Any, indent: int = 0) -> str:
        if isinstance(data, dict):
            lines: list[str] = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{' ' * indent}{key}:")
                    lines.append(_dump(value, indent + 2).rstrip("\n"))
                else:
                    lines.append(f"{' ' * indent}{key}: {_dump_scalar(value)}")
            return "\n".join(lines) + "\n"
        if isinstance(data, list):
            lines = []
            for item in data:
                if isinstance(item, dict) and item:
                    first = True
                    for key, value in item.items():
                        if first:
                            lines.append(f"{' ' * indent}- {key}: {_dump_scalar(value)}")
                            first = False
                        else:
                            lines.append(f"{' ' * (indent + 2)}{key}: {_dump_scalar(value)}")
                else:
                    lines.append(f"{' ' * indent}- {_dump_scalar(item)}")
            return "\n".join(lines) + "\n"
        return f"{' ' * indent}{_dump_scalar(data)}\n"

    def safe_dump(data: Any, sort_keys: bool = False) -> str:
        _ = sort_keys
        return _dump(data)

    mod.safe_load = safe_load
    mod.safe_dump = safe_dump
    sys.modules["yaml"] = mod


def _install_mcp() -> None:
    if "mcp.server.fastmcp" in sys.modules:
        return
    mcp = ModuleType("mcp")
    server = ModuleType("mcp.server")
    fast = ModuleType("mcp.server.fastmcp")

    class FastMCP:
        def __init__(self, name: str) -> None:
            self.name = name

        def tool(self):
            def deco(func):
                return func

            return deco

    fast.FastMCP = FastMCP
    sys.modules.update({"mcp": mcp, "mcp.server": server, "mcp.server.fastmcp": fast})


def _install_llama_and_tqdm() -> None:
    if "llama_index.core.embeddings" not in sys.modules:
        root = ModuleType("llama_index")
        core = ModuleType("llama_index.core")
        emb = ModuleType("llama_index.core.embeddings")
        schema = ModuleType("llama_index.core.schema")

        class BaseEmbedding:
            def __init__(self, **kwargs: Any) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class TextNode:
            def __init__(self, text: str, id_: str, metadata: dict[str, Any]) -> None:
                self.text = text
                self.id_ = id_
                self.metadata = metadata
                self.embedding: list[float] | None = None

        emb.BaseEmbedding = BaseEmbedding
        schema.TextNode = TextNode
        core.embeddings = emb
        core.schema = schema
        root.core = core
        sys.modules.update(
            {
                "llama_index": root,
                "llama_index.core": core,
                "llama_index.core.embeddings": emb,
                "llama_index.core.schema": schema,
            }
        )

    if "tqdm" not in sys.modules:
        tmod = ModuleType("tqdm")

        class _TQDM:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                _ = args, kwargs

            def update(self, _n: int = 1) -> None:
                return None

            def close(self) -> None:
                return None

        tmod.tqdm = _TQDM
        sys.modules["tqdm"] = tmod


def _install_faiss_lib() -> None:
    if "faiss_storage_lib.engine.faiss_engine" in sys.modules:
        return
    root = ModuleType("faiss_storage_lib")
    core = ModuleType("faiss_storage_lib.core")
    schema = ModuleType("faiss_storage_lib.core.schema")
    engine_pkg = ModuleType("faiss_storage_lib.engine")
    engine_mod = ModuleType("faiss_storage_lib.engine.faiss_engine")

    class VectorDocument:
        def __init__(self, uid: str, vector: list[float], payload: dict[str, Any]) -> None:
            self.uid = uid
            self.vector = vector
            self.payload = payload
            self.score = 1.0

    class FaissEngine:
        def __init__(self, _path: str, _dim: int) -> None:
            self.docs: list[VectorDocument] = []
            self.persisted = False
            self.deleted: list[tuple[str, Any]] = []

        def add(self, docs: list[VectorDocument]) -> None:
            self.docs.extend(docs)

        def persist(self) -> None:
            self.persisted = True

        def search(self, _embedding: list[float], top_k: int) -> list[VectorDocument]:
            return self.docs[:top_k]

        def get_by_id(self, uid: str) -> VectorDocument | None:
            for doc in self.docs:
                if doc.uid == uid:
                    return doc
            return None

        def delete_by_metadata(self, key: str, value: Any) -> None:
            self.deleted.append((key, value))
            self.docs = [d for d in self.docs if d.payload.get("metadata", {}).get(key) != value]

    schema.VectorDocument = VectorDocument
    engine_mod.FaissEngine = FaissEngine
    sys.modules.update(
        {
            "faiss_storage_lib": root,
            "faiss_storage_lib.core": core,
            "faiss_storage_lib.core.schema": schema,
            "faiss_storage_lib.engine": engine_pkg,
            "faiss_storage_lib.engine.faiss_engine": engine_mod,
        }
    )


_install_requests()
_install_yaml()
_install_mcp()
_install_llama_and_tqdm()
_install_faiss_lib()
