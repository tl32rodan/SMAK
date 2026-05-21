# SMAK Demo — Semantic Mesh Augmented Kernel

This demo shows SMAK's core value proposition: **one query surfaces not just the most
relevant code, but every linked artifact — bugs, tests, and docs — through the semantic
mesh of sidecar relations.**

A fully self-contained workspace is included:

| Workspace | Project |
|-----------|---------|
| `workspace_a` | CSV Editor utility |

The workspace comes with **pre-populated sidecar files** that wire code symbols to their
related issues, so the mesh traversal works immediately after ingestion.

---

## Layout

```
demo/
└── workspace_a/
    ├── workspace_config.yaml   # indices: source_code, issues, tests, documentation
    ├── src/
    │   ├── csv_editor.py
    │   ├── .csv_editor.py.sidecar.yaml   # pre-populated intent + relations
    │   └── tests/
    │       └── test_csv_editor.py
    ├── documentation/
    │   └── csv-editor-usage.md
    └── issues/
        └── csv-editor-known-issues.md
```

The `path:` field on each index entry in `workspace_config.yaml` controls which directory
`smak ingest` reads. For example, `source_code` has `path: ./src`, so `smak ingest
--index source_code` ingests from `<workspace>/src/` automatically.

Every index **must** specify a `uri` field — an **absolute path or `$VAR` path** to the
vector store. Variables are defined in the `env:` block of `workspace_config.yaml`
(not shell env vars). The demo defines `SMAK_DATA` in its `env:` block.

---

## Prerequisites

**1. Install SMAK**

```bash
pip install -e .
```

**2. Embedding server**

SMAK uses an embedding model served via an OpenAI-compatible API.
To override the default endpoint or model, edit `src/smak/embedding_setup.yaml`.

Verify the server is reachable before running any `ingest` or `query` command.

---

## CLI walkthrough

All commands below are run from the **repository root**.

### Step 1: Ingest source code

```bash
smak ingest --index source_code \
            --config demo/workspace_a/workspace_config.yaml \
            --workers 1
```

Expected output:

```
Starting ingestion for '.../workspace_a/src' -> Index: 'source_code'...
Ingestion Complete!
   - Processed Files: 2
   - Skipped Files: 0
   - Vectors Added: 5
   - Ghost Files Pruned: 0
```

The Python parser extracts symbols from `csv_editor.py` and `test_csv_editor.py`.
The sidecar file `.csv_editor.py.sidecar.yaml` is picked up automatically — it enriches
each symbol with `intent` text and `relations` pointers.

### Step 2: Ingest issues

```bash
smak ingest --index issues \
            --config demo/workspace_a/workspace_config.yaml \
            --workers 1
```

This loads `csv-editor-known-issues.md` into the `issues` index.
Its UID becomes `csv-editor-known-issues` (taken from the frontmatter `symbol:` field).

### Step 3: Ingest documentation

```bash
smak ingest --index documentation \
            --config demo/workspace_a/workspace_config.yaml \
            --workers 1
```

### Step 4: Inspect symbols

See which UIDs were extracted from the source file:

```bash
smak sidecar inspect demo/workspace_a/src/csv_editor.py
```

Expected output:

```
CsvEditor
CsvEditor.append_row
CsvEditor.update_cell
CsvEditor.read_rows
```

These names match the `name:` entries in `.csv_editor.py.sidecar.yaml` exactly.

### Step 5: The semantic mesh query

```bash
smak query "why does update_cell raise an error" \
      --index source_code \
      --top-k 3 \
      --config demo/workspace_a/workspace_config.yaml
```

Example output (abbreviated):

```json
{
  "hits": [
    {
      "uid": "src/csv_editor.py::CsvEditor.update_cell",
      "match_type": "semantic",
      "score": 0.91,
      "content": "def update_cell(self, row_index, column_index, value): ..."
    }
  ],
  "related_context": [
    {
      "uid": "csv-editor-known-issues",
      "match_type": "relation",
      "source_hit": "src/csv_editor.py::CsvEditor.update_cell",
      "content": "# CSV Editor Known Issues\n- ISSUE-001: update_cell raises IndexError ..."
    }
  ]
}
```

**What just happened:** SMAK found `update_cell` semantically, then resolved its sidecar
relation (`csv-editor-known-issues`) and fetched the full issue content from the `issues`
index — all in one query, without you knowing the issue file existed.

### Step 6: Validate mesh integrity

```bash
smak doctor --config demo/workspace_a/workspace_config.yaml
```

Expected output:

```
Mesh diagnostics passed.
```

---

## MCP server walkthrough

Start the server (from repository root, keep it running in a separate terminal):

```bash
python -m smak.mcp_server
```

The server starts stateless. Every tool call includes a `config` parameter pointing to
a `workspace_config.yaml` file.

### list_available_indices

```json
// Tool call
{ "name": "list_available_indices", "arguments": { "config": "demo/workspace_a/workspace_config.yaml" } }

// Response
[
  {"name": "source_code", "description": "Contains the project's source code..."},
  {"name": "issues", "description": "Contains historical bug reports..."},
  ...
]
```

### refresh_knowledge — ingest source_code

```json
{
  "name": "refresh_knowledge",
  "arguments": {
    "config": "demo/workspace_a/workspace_config.yaml",
    "index": "source_code"
  }
}

// Response
"Ingestion Complete! Processed Files: 2, Skipped Files: 0, Vectors Added: 5"
```

### semantic_search — the mesh traversal

```json
{
  "name": "semantic_search",
  "arguments": {
    "config": "demo/workspace_a/workspace_config.yaml",
    "query": "why does update_cell raise an error",
    "index": "source_code",
    "top_k": 3
  }
}
```

### multi_index_search — search all indices at once

```json
{
  "name": "multi_index_search",
  "arguments": {
    "config": "demo/workspace_a/workspace_config.yaml",
    "query": "error handling",
    "top_k": 3
  }
}
```

### workspace_status — health dashboard

```json
{ "name": "workspace_status", "arguments": { "config": "demo/workspace_a/workspace_config.yaml" } }
```

### validate_mesh

```json
{ "name": "validate_mesh", "arguments": { "config": "demo/workspace_a/workspace_config.yaml" } }
```

---

## Sidecar format reference

Sidecar files live next to their source file with the suffix `.sidecar.yaml`:

```
src/csv_editor.py
src/.csv_editor.py.sidecar.yaml   ← sidecar
```

Format:

```yaml
symbols:
  - name: ClassName               # must match symbol name from `smak sidecar inspect`
    intent: "Human-readable intent description used to enrich embeddings."
    relations:
      - "uid-of-related-item"     # full UID as stored in any index
  - name: ClassName.method_name
    intent: "..."
    relations: []
```

**Class-level relations are inherited by all methods of that class.**

The `relations` list accepts any UID from any index in the same workspace —
source code UIDs (`src/file.py::Symbol`), issue UIDs (from frontmatter `symbol:` or
slugified header), or documentation UIDs. Cross-index relation traversal is automatic
during `smak query` and `semantic_search`.
