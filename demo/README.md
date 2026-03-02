# SMAK Demo — Semantic Mesh Augmented Kernel

This demo shows SMAK's core value proposition: **one query surfaces not just the most
relevant code, but every linked artifact — bugs, tests, and docs — through the semantic
mesh of sidecar relations.**

Two fully self-contained workspaces are included:

| Workspace | Project | Registry name |
|-----------|---------|---------------|
| `workspace_a` | CSV Editor utility | `demo_flow_a` |
| `workspace_b` | Log Analyzer utility | `demo_flow_b` |

Both workspaces come with **pre-populated sidecar files** that wire code symbols to their
related issues, so the mesh traversal works immediately after ingestion.

---

## Layout

```
demo/
├── all_workspaces.yaml          # MCP server registry
├── workspace_a/
│   ├── workspace_config.yaml   # 4 indices: source_code, issues, tests, documentation
│   ├── src/
│   │   ├── csv_editor.py
│   │   ├── csv_editor.py.sidecar.yaml   # pre-populated intent + relations
│   │   └── tests/
│   │       └── test_csv_editor.py
│   ├── documentation/
│   │   └── csv-editor-usage.md
│   └── issues/
│       └── csv-editor-known-issues.md
└── workspace_b/
    ├── workspace_config.yaml   # 4 indices: source_code, issues, tests, documentation
    ├── src/
    │   ├── log_analyzer.py
    │   ├── log_analyzer.py.sidecar.yaml  # pre-populated intent + relations
    │   └── tests/
    │       └── test_log_analyzer.py
    ├── documentation/
    │   └── log-analyzer-design.md
    └── issues/
        └── log-analyzer-known-issues.md
```

The `path:` field on each index entry in `workspace_config.yaml` controls which directory
`smak ingest` reads. For example, `source_code` has `path: ./src`, so `smak ingest
--index source_code` ingests from `<workspace>/src/` automatically.

---

## Prerequisites

**1. Install SMAK**

```bash
pip install -e .
```

**2. Embedding server**

SMAK uses a Nomic embedding model served via Ollama or a compatible OpenAI-style API.
Set the endpoint if it differs from the default:

```bash
export SMAK_NOMIC_API_BASE="http://localhost:11436"   # default: http://f15dtpai1:11436
export SMAK_NOMIC_MODEL="nomic-embed-text:latest"      # default: nomic_embed_text:latest
```

Verify the server is reachable before running any `ingest` or `query` command.

---

## Part 1 — workspace_a: CSV Editor

### 1.1 CLI walkthrough

All commands below are run from the **repository root**.

#### Step 1: Ingest source code

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
```

The Python parser extracts symbols from `csv_editor.py` and `test_csv_editor.py`.
The sidecar file `csv_editor.py.sidecar.yaml` is picked up automatically — it enriches
each symbol with `intent` text and `relations` pointers.

#### Step 2: Ingest issues

```bash
smak ingest --index issues \
            --config demo/workspace_a/workspace_config.yaml \
            --workers 1
```

This loads `csv-editor-known-issues.md` into the `issues` index.
Its UID becomes `csv-editor-known-issues` (taken from the frontmatter `symbol:` field).

#### Step 3: Ingest documentation

```bash
smak ingest --index documentation \
            --config demo/workspace_a/workspace_config.yaml \
            --workers 1
```

#### Step 4: Inspect symbols

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

These names match the `name:` entries in `csv_editor.py.sidecar.yaml` exactly.

#### Step 5: The semantic mesh query

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

#### Step 6: Update a sidecar entry (optional)

Add or edit relations using `smak sidecar update`.
First inspect to confirm symbol names:

```bash
smak sidecar inspect demo/workspace_a/src/csv_editor.py
```

Then update:

```bash
smak sidecar update demo/workspace_a/src/csv_editor.py \
     --updates '[
       {
         "symbol": "CsvEditor.append_row",
         "intent": "Appends a row. Thread-unsafe on concurrent writes.",
         "relations": []
       }
     ]'
```

Re-ingest to propagate the change into the vector store:

```bash
smak ingest --index source_code \
            --config demo/workspace_a/workspace_config.yaml \
            --workers 1
```

#### Step 7: Validate mesh integrity

```bash
smak doctor --config demo/workspace_a/workspace_config.yaml
```

Expected output:

```
Mesh diagnostics passed.
```

The doctor checks that every relation UID referenced in a sidecar actually exists in at
least one vector index. Run this after deleting or renaming files to catch dangling links.

---

### 1.2 MCP server walkthrough

Start the server (from repository root, keep it running in a separate terminal):

```bash
python -m smak.mcp_server --registry ./demo/all_workspaces.yaml
```

The following examples show each MCP tool call and its expected response.

#### list_available_workspaces

```json
// Tool call
{ "name": "list_available_workspaces", "arguments": {} }

// Response
{
  "demo_flow_a": { "path": "./workspace_a", "description": "Demo workspace A for primary flow." },
  "demo_flow_b": { "path": "./workspace_b", "description": "Demo workspace B for secondary flow." }
}
```

#### refresh_knowledge — ingest source_code

```json
// Tool call
{
  "name": "refresh_knowledge",
  "arguments": {
    "workspace": "demo_flow_a",
    "folder": "./src",
    "index": "source_code"
  }
}

// Response
"Ingestion Complete! Processed Files: 2, Skipped Files: 0, Vectors Added: 5"
```

#### refresh_knowledge — ingest issues

```json
{
  "name": "refresh_knowledge",
  "arguments": {
    "workspace": "demo_flow_a",
    "folder": "./issues",
    "index": "issues"
  }
}
```

#### semantic_search — the mesh traversal

```json
// Tool call
{
  "name": "semantic_search",
  "arguments": {
    "workspace": "demo_flow_a",
    "query": "why does update_cell raise an error",
    "index": "source_code",
    "top_k": 3
  }
}

// Response
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

#### manage_sidecar — inspect symbols

```json
// Tool call
{
  "name": "manage_sidecar",
  "arguments": {
    "workspace": "demo_flow_a",
    "action": "inspect",
    "file_path": "./src/csv_editor.py"
  }
}

// Response
["CsvEditor", "CsvEditor.append_row", "CsvEditor.update_cell", "CsvEditor.read_rows"]
```

#### manage_sidecar — update intent and relations

```json
{
  "name": "manage_sidecar",
  "arguments": {
    "workspace": "demo_flow_a",
    "action": "update",
    "file_path": "./src/csv_editor.py",
    "updates": [
      {
        "symbol": "CsvEditor.read_rows",
        "intent": "Reads all rows. Returns empty list for an empty file.",
        "relations": []
      }
    ]
  }
}
```

#### validate_mesh

```json
// Tool call
{ "name": "validate_mesh", "arguments": { "workspace": "demo_flow_a" } }

// Response
"Mesh diagnostics passed."
```

---

## Part 2 — workspace_b: Log Analyzer

`workspace_b` is a completely separate project (log file analysis) to prove that
workspace isolation works: a query in `demo_flow_b` never surfaces CSV editor content.

### 2.1 CLI walkthrough

#### Step 1: Ingest source code

```bash
smak ingest --index source_code \
            --config demo/workspace_b/workspace_config.yaml \
            --workers 1
```

Expected output:

```
Starting ingestion for '.../workspace_b/src' -> Index: 'source_code'...
Ingestion Complete!
   - Processed Files: 2
   - Skipped Files: 0
   - Vectors Added: 6
```

The sidecar `log_analyzer.py.sidecar.yaml` is picked up automatically, linking
`LogAnalyzer.parse` and `LogAnalyzer` to the `log-parse-error` issue UID.

#### Step 2: Ingest issues

```bash
smak ingest --index issues \
            --config demo/workspace_b/workspace_config.yaml \
            --workers 1
```

The issue file's frontmatter `symbol: log-parse-error` sets its UID to `log-parse-error`.

#### Step 3: Ingest documentation

```bash
smak ingest --index documentation \
            --config demo/workspace_b/workspace_config.yaml \
            --workers 1
```

#### Step 4: Inspect symbols

```bash
smak sidecar inspect demo/workspace_b/src/log_analyzer.py
```

Expected output:

```
LogEntry
LogAnalyzer
LogAnalyzer.parse
LogAnalyzer.count_by_level
LogAnalyzer.filter_by_level
```

#### Step 5: Semantic mesh query

```bash
smak query "how are log entries parsed" \
      --index source_code \
      --top-k 3 \
      --config demo/workspace_b/workspace_config.yaml
```

Example output:

```json
{
  "hits": [
    {
      "uid": "src/log_analyzer.py::LogAnalyzer.parse",
      "match_type": "semantic",
      "score": 0.94,
      "content": "def parse(self) -> list[LogEntry]: ..."
    }
  ],
  "related_context": [
    {
      "uid": "log-parse-error",
      "match_type": "relation",
      "source_hit": "src/log_analyzer.py::LogAnalyzer.parse",
      "content": "# Log Analyzer Known Issues\n- ISSUE-101: parse() silently skips malformed lines ..."
    }
  ]
}
```

#### Step 6: Update a sidecar entry (optional)

```bash
smak sidecar update demo/workspace_b/src/log_analyzer.py \
     --updates '[
       {
         "symbol": "LogAnalyzer.count_by_level",
         "intent": "Frequency map by level. Note: case-sensitive keys (see ISSUE-102).",
         "relations": ["log-parse-error"]
       }
     ]'
```

Re-ingest after updating:

```bash
smak ingest --index source_code \
            --config demo/workspace_b/workspace_config.yaml \
            --workers 1
```

#### Step 7: Validate mesh

```bash
smak doctor --config demo/workspace_b/workspace_config.yaml
```

---

### 2.2 MCP server walkthrough

With the MCP server still running (`python -m smak.mcp_server --registry ./demo/all_workspaces.yaml`):

#### refresh_knowledge — ingest source_code

```json
{
  "name": "refresh_knowledge",
  "arguments": {
    "workspace": "demo_flow_b",
    "folder": "./src",
    "index": "source_code"
  }
}
```

#### refresh_knowledge — ingest issues

```json
{
  "name": "refresh_knowledge",
  "arguments": {
    "workspace": "demo_flow_b",
    "folder": "./issues",
    "index": "issues"
  }
}
```

#### semantic_search

```json
// Tool call
{
  "name": "semantic_search",
  "arguments": {
    "workspace": "demo_flow_b",
    "query": "how are log entries parsed",
    "index": "source_code",
    "top_k": 3
  }
}

// Response shows log analyzer content — zero CSV editor content
{
  "hits": [
    {
      "uid": "src/log_analyzer.py::LogAnalyzer.parse",
      "match_type": "semantic",
      "score": 0.94,
      "content": "def parse(self) -> list[LogEntry]: ..."
    }
  ],
  "related_context": [
    {
      "uid": "log-parse-error",
      "match_type": "relation",
      "source_hit": "src/log_analyzer.py::LogAnalyzer.parse",
      "content": "# Log Analyzer Known Issues\n- ISSUE-101: ..."
    }
  ]
}
```

#### manage_sidecar — inspect

```json
{
  "name": "manage_sidecar",
  "arguments": {
    "workspace": "demo_flow_b",
    "action": "inspect",
    "file_path": "./src/log_analyzer.py"
  }
}
```

#### validate_mesh

```json
{ "name": "validate_mesh", "arguments": { "workspace": "demo_flow_b" } }
```

---

## Part 3 — Workspace isolation

Run the same query against both workspaces to confirm isolation:

**CLI:**

```bash
smak query "parse error handling" \
      --index source_code --top-k 2 \
      --config demo/workspace_a/workspace_config.yaml

smak query "parse error handling" \
      --index source_code --top-k 2 \
      --config demo/workspace_b/workspace_config.yaml
```

The first returns CSV editor content; the second returns log analyzer content.
Vector stores live in separate `smak_data/` directories inside each workspace root.

**MCP (same server, different `workspace` argument):**

```json
{ "name": "semantic_search", "arguments": { "workspace": "demo_flow_a", "query": "parse error handling", "index": "source_code" } }
{ "name": "semantic_search", "arguments": { "workspace": "demo_flow_b", "query": "parse error handling", "index": "source_code" } }
```

---

## Sidecar format reference

Sidecar files live next to their source file with the suffix `.sidecar.yaml`:

```
src/csv_editor.py
src/csv_editor.py.sidecar.yaml   ← sidecar
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
