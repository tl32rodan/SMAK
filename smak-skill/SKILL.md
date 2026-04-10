---
name: smak-skill
description: SMAK (Semantic Mesh Augmented Kernel) - A semantic search and context expansion tool. Use ONLY when you do NOT already know the file path or symbol name. If you know the target, use grep or read the file directly. Use this exclusively to explore code intent, historical context, and 1-hop bi-directional cross-entity relations (e.g., linking code with issues, tests, and docs in both directions).
---

# SMAK Skill

## 1. WHAT SMAK IS

SMAK (**Semantic Mesh Augmented Kernel**) is a **passive MCP knowledge kernel** — a read/write layer for:
- **Semantic retrieval**: embedding-based search over code, issues, tests, and docs.
- **Sidecar metadata**: per-file YAML storing human-written `intent` and cross-entity `relations` for each code symbol.
- **1-hop mesh traversal**: when a search hit has sidecar relations, SMAK auto-fetches the linked entities and returns them alongside the hit.

### Key concepts

| Concept | Description |
|---|---|
| **Vector store** | Embeddings of every ingested code unit. Updated by `ingest`. Queried by `search`. |
| **Sidecar file** | Hidden YAML (e.g. `src/.foo.py.sidecar.yaml`) storing `intent` and `relations` per symbol. Updated by `enrich_symbol` / `enrich_file`. Read at query time for 1-hop expansion. |
| **UID** | Globally unique identifier: `{path}::{symbol}` (e.g. `/home/user/project/src/foo.py::ClassName.method` or `$DDI_ROOT_PATH/src/foo.py::ClassName.method`). |
| **Symbol name** | Short name without path prefix (e.g. `ClassName.method`). Used in `enrich_symbol`. |
| **path_env** | Optional config field mapping UIDs to environment variables instead of absolute paths. |

### Two independent data stores — know the difference

```
┌─────────────────────────────────────────────────────────────┐
│ Vector store (embeddings)        Sidecar files (YAML)       │
│ ─────────────────────────        ────────────────────────   │
│ Written by: ingest               Written by: enrich_*       │
│ Read by:    search, lookup       Read by:    search          │
│                                              (at query time) │
│                                                             │
│ Enriching a sidecar does NOT update the vector store.       │
│ You must call ingest to re-embed.                           │
└─────────────────────────────────────────────────────────────┘
```

### Config structure

```yaml
# workspace_config.yaml — agent passes path dynamically per tool call
indices:
  - name: source_code
    description: "RTL Verilog modules for DDR5 PHY datapath"
    paths: [$DDI_ROOT_PATH/src]
    uri: $SMAK_DATA/source_code          # absolute or $ENV_VAR — never relative
    path_env: DDI_ROOT_PATH
  - name: issues
    description: "Jira tickets and postmortems for timing closure failures"
    paths: [./issues]
    uri: $SMAK_DATA/issues
```

Every SMAK tool takes `config` as its first parameter — the path to `workspace_config.yaml`.
Indices are **not limited to any default set** — define any number with any names.

**`uri` must be an absolute path or use an environment variable** (e.g. `$SMAK_DATA/source_code`).
Relative paths like `./smak_data/...` are rejected. This ensures portability — if data moves,
only the env var (or absolute path) needs to change, with no ambiguity about the base directory.

---

## 2. WHEN TO USE / NOT USE

### MANDATORY PRE-CHECK — before every `search` call

**Do you already know the file path or function/class name?**
- If **YES** → **DO NOT** call `search`. Use `grep`/`rg` or read the file directly.
- If **NO** → proceed with `search`.

### Use SMAK for
- **Intent discovery**: understand the "why" behind code — when you don't know where it lives.
- **1-hop context expansion**: from a code hit, auto-fetch linked issues/docs/tests.
- **Sidecar enrichment**: annotate symbols with intent and relations via `enrich_symbol`.
- **Cross-index exploration**: use `search_all` to find related entities across all indices.

### Do NOT use SMAK for
- **Exact string matching** → use `rg` / `grep`.
- **Go-to-definition** → use LSP / IDE navigation.
- **When you already know the file path or symbol name** → read the file directly.

### Anti-hallucination stop rule
If results are low-relevance **2 times in a row**, **STOP**. Ask user for a narrower starting point.

---

## 3. MCP TOOL REFERENCE (10 tools)

Every tool takes `config` (path to `workspace_config.yaml`) as first parameter.

### Discovery
- **`describe_workspace(config)`** — list all indices with names, descriptions, paths. Call this first.

### Search
- **`search(config, query, index, top_k=5)`** — semantic search within one index
- **`search_all(config, query, indices=None, top_k=3)`** — search across all (or specified) indices at once
- **`lookup(config, uid, index)`** — verify a UID exists in the vector store

### Sidecar enrichment
- **`enrich_symbol(config, file_path, symbol, intent?, relations?, index, bidirectional=False)`** — annotate one symbol (auto-syncs sidecar, auto-clears stale symbols, validates symbol exists). When `bidirectional=True`, also adds reverse relations from targets back to this symbol.
- **`enrich_file(config, file_path, index)`** — create/sync a file's sidecar (stub entries for all symbols)
- **`enrich_batch(config, file_paths, index)`** — sync sidecars for multiple files at once

### Maintenance
- **`ingest(config, index, follow_symlinks=True)`** — re-embed files into vector store (**resource-intensive**)
- **`check_health(config)`** — run integrity diagnostics, returns `{status, issues}`
- **`graph_stats(config)`** — knowledge graph coverage statistics: total/enriched symbols, relations, coverage %, per-index breakdown, asymmetric relation warnings

---

## 4. QUERY FORMULATION

Write queries as **natural-language descriptions of behavior or purpose** — not symbol names or file paths.

### Good vs bad queries
```
Good: "CSV file row append logic"
Good: "error handling for out-of-range index in cell update"
Bad:  "append_row"         ← grep instead
Bad:  "csv_editor.py"      ← file tools instead
```

When unsure which index, use `search_all` to search everything at once.

### Search result format
```json
{
  "hits": [{
    "uid": "...",
    "exact_relative_path": "src/csv_editor.py",
    "match_type": "semantic",
    "score": 0.89,
    "content": "..."
  }],
  "related_context": [{
    "uid": "...",
    "match_type": "relation",
    "source_hit": "...",
    "content": "..."
  }]
}
```

**Field rules:**
- `exact_relative_path` — copy EXACTLY as `file_path` for `enrich_*` tools. Never rewrite.
- `uid` — use in `relations` lists and with `lookup`.

---

## 5. SIDECAR ENRICHMENT

### Annotate a symbol (the primary workflow)

```python
# One call does everything: validate symbol, sync sidecar, write enrichment
enrich_symbol(
  config="./workspace_config.yaml",
  file_path="src/csv_editor.py",       # from search hit's exact_relative_path
  symbol="CsvEditor.update_cell",       # short name
  intent="Rewrites entire file to update one cell. Known IndexError issue.",
  relations=["$DDI_ROOT_PATH/issues/csv-bugs.md::*"],
  index="source_code"
)
```

What `enrich_symbol` does internally:
1. Validates `symbol` exists in the file (returns error + valid list if not)
2. Syncs the sidecar (creates if missing, auto-clears stale symbols)
3. Writes intent and relations

### Initialize sidecars for a directory

```python
# Create stub sidecars for all files
enrich_batch(
  config="./workspace_config.yaml",
  file_paths=["src/a.py", "src/b.py", "src/c.py"],
  index="source_code"
)
```

### Sidecar YAML format (for reference)
```yaml
# src/.csv_editor.py.sidecar.yaml
symbols:
  - name: CsvEditor                    # short symbol name
    intent: "Manages CSV read/write"
    relations:
      - "$DDI_ROOT_PATH/issues/known-issues.md::*"    # full UID
  - name: CsvEditor.update_cell
    intent: ""
    relations: []
```

---

## 6. COMMON PIPELINES

### Pipeline A: Link code to related issues

```python
cfg = "./workspace_config.yaml"

# 1. Find the code
hit = search(config=cfg, query="CSV cell update logic", index="source_code")

# 2. Find the related issue
issue = search(config=cfg, query="cell update out of range bug", index="issues")

# 3. Verify the issue UID exists
lookup(config=cfg, uid=issue_uid, index="issues")

# 4. Annotate the code symbol with the relation
enrich_symbol(
  config=cfg,
  file_path=hit["exact_relative_path"],
  symbol="CsvEditor.update_cell",
  relations=[issue_uid],
  index="source_code"
)

# 5. (Optional) Add reverse relation
enrich_symbol(
  config=cfg,
  file_path=issue_path,
  symbol="*",
  relations=[code_uid],
  index="issues"
)
```

### Pipeline B: Broad exploration

```python
# Don't know which index? Search everything.
search_all(config=cfg, query="authentication timeout handling", top_k=3)
```

### Pipeline C: Health check

```python
check_health(config=cfg)
# → {"status": "healthy", "issues": []}
# → {"status": "unhealthy", "issues": ["Orphaned sidecar: ..."]}
```

---

## 7. INDEX DESIGN PATTERNS

Indices are **arbitrary** — not limited to any default set.

```yaml
# EDA project example
indices:
  - name: rtl_code
    description: "Verilog/SystemVerilog RTL modules for DDR5 PHY datapath"
    paths: [$DDI_ROOT_PATH/rtl/phy]
    uri: $SMAK_DATA/rtl_code
    path_env: DDI_ROOT_PATH
  - name: verification
    description: "UVM testbenches and coverage models"
    paths: [$DDI_ROOT_PATH/verif]
    uri: $SMAK_DATA/verification
    path_env: DDI_ROOT_PATH
  - name: release_notes
    description: "Release notes, known issues, and ECO history"
    paths: [./release_notes]
    uri: $SMAK_DATA/release_notes
```

### Writing effective descriptions

The `description` field is the **agent's ONLY hint** for index selection.

- Include **file types** (Verilog, Python, Markdown)
- Include **domain terms** (DDR5, PHY, authentication)
- Describe **what questions** this index answers
- Be specific: `"RTL Verilog modules for DDR5 PHY datapath"` > `"source code"`

---

## 8. ENVIRONMENT VARIABLE UIDs

Use `path_env` when your codebase lives at different absolute paths:

```yaml
indices:
  - name: source_code
    paths: [$DDI_ROOT_PATH/src]
    uri: $SMAK_DATA/source_code
    path_env: DDI_ROOT_PATH
```

UIDs become `$DDI_ROOT_PATH/src/a.py::ClassName` instead of absolute paths.
At query time, `$DDI_ROOT_PATH` is expanded to the current environment value.

Path mismatch warnings are emitted when editing sidecars in SOS workspaces — this is expected.

For CliosoftSOS / EDA environments, see **[`sos-smak-skill/SKILL.md`](../sos-smak-skill/SKILL.md)** for the full three-layer path model (online → version control → workspace) and operational workflows.

---

## 9. STRICT RULES

1. **Never use `search` when you already know the target.**
2. **`file_path` must exactly match `exact_relative_path`** from search hits.
3. **`symbol` = short name** (e.g. `CsvEditor.update_cell`), not full UID.
4. **`relations` = full UIDs** (e.g. `/abs/path::Symbol` or `$ENV_VAR/path::Symbol`).
5. **Always `lookup`** to verify a UID exists before adding it to relations.
6. **Sidecar updates ≠ vector store updates.** Call `ingest` only when source files change.
7. **`ingest` is resource-intensive.** Don't call casually.
8. **`uri` must be absolute or use `$ENV_VAR`.** Relative URIs are rejected. Use an environment variable (e.g. `$SMAK_DATA/index_name`) or a full absolute path.
