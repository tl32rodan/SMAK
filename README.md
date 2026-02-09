# SMAK

SMAK now stands for **Semantic Mesh Augmented Kernel** and is focused as a **passive MCP knowledge kernel**.

It is the source-of-truth layer for:
- code symbols,
- sidecar intent metadata,
- issue/document relations,
- and semantic retrieval context.

It does **not** run autonomous review/agent loops itself. Those workflows belong to MCP clients (IDE agents, CI skills, etc.) that call SMAK tools.

## What SMAK provides

### 1) Ingestion kernel
SMAK ingests files into knowledge units, enriches them with sidecar metadata, computes embeddings, and stores vectors.

Key behavior:
- **Canonical symbol IDs**: `relative/path.py::Symbol`.
- **Context inheritance**: class-level sidecar relations are inherited by class methods.
- **Idempotent re-ingest**: existing vectors for a source are deleted before writing fresh vectors.

### 2) MCP-facing API surface
`src/smak/mcp_server.py` provides core tool methods:
- `get_file_structure(file_path)`
- `get_symbol_context(file_path, symbol)`
- `upsert_sidecar(file_path, symbol, intent=None, relations=None)`
- `link_issue(symbol_id, issue_id)`
- `diagnose_mesh(path=None)`

### 3) CLI utilities
- `smak init`
- `smak ingest`
- `smak search`
- `smak sidecar init`
- `smak doctor`

---

## Quick start

### Generate config
```bash
smak init --path workspace_config.yaml
```

### Ingest a folder
```bash
smak ingest --folder ./src --index source_code --config workspace_config.yaml
```

### List canonical symbols from a file
```bash
smak search ./src/csv_editor.py --config workspace_config.yaml
# e.g. src/auth.py::Auth
#      src/auth.py::Auth.login
```

### Generate sidecar skeleton
```bash
smak sidecar init ./src/csv_editor.py --config workspace_config.yaml
```

### Run mesh diagnostics
```bash
smak doctor --path .
```
