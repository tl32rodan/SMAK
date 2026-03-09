---
name: smak-skill
description: SMAK (Semantic Mesh Augmented Kernel) - A semantic search and context expansion tool. Use this exclusively to explore code intent, historical context, and 1-hop bi-directional cross-entity relations (e.g., linking code with issues, tests, and docs in both directions).
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
| **Vector store** | Stores embeddings of every ingested code unit. Updated by `refresh_knowledge` (ingestion). Queried by `semantic_search`. |
| **Sidecar file** | Hidden YAML (e.g. `src/.foo.py.sidecar.yaml`) storing `intent` and `relations` per symbol. Updated by `update_sidecar`. Read at query time for 1-hop expansion. |
| **UID** | Globally unique identifier for a vector-store entry: `{absolute_path}::{symbol}` (e.g. `/home/user/project/src/foo.py::ClassName.method`). |
| **Symbol name** | Short name without path prefix (e.g. `ClassName.method`). Used in sidecar `name` fields and as the `symbol` parameter in sidecar tools. |

### Two independent data stores — know the difference

```
┌─────────────────────────────────────────────────────────────┐
│ Vector store (embeddings)        Sidecar files (YAML)       │
│ ─────────────────────────        ────────────────────────   │
│ Written by: refresh_knowledge    Written by: update_sidecar │
│ Read by:    semantic_search      Read by:    semantic_search │
│             lookup_symbol                    (at query time) │
│                                                             │
│ Updating a sidecar does NOT update the vector store.        │
│ You must call refresh_knowledge to re-ingest.               │
└─────────────────────────────────────────────────────────────┘
```

**Critical rule**: `update_sidecar` only writes `.sidecar.yaml` files on disk. It does **not** update vectors. If you need the vector store to reflect new or changed source files, you must call `refresh_knowledge` — but be aware this is **resource-intensive** (re-embeds all files in the index). Do not call it casually.

### Config hierarchy

```
registry.yaml                          ← pass --registry to MCP server
  └── configs:
        <config_name>:
          config_path: ./workspace/workspace_config.yaml
            └── indices:
                  - name: source_code   path: ./src
                  - name: issues        path: ./issues
                  - name: tests         path: ./src/tests
                  - name: documentation path: ./documentation
```

- `list_available_configs()` → valid `config` values.
- `list_available_indices(config)` → valid `index` values for a config.

---

## 2. WHEN TO USE / NOT USE

### Use SMAK for
- **Intent discovery**: understand the "why" behind hacks, tradeoffs, or legacy behavior.
- **1-hop context expansion**: from a code hit, auto-fetch linked issues/docs/tests via relations.
- **Sidecar lifecycle**: inspect, create, or update `intent` + `relations` metadata for source files.

### Do NOT use SMAK for
- **Exact string matching** → use `rg` / `grep`.
- **Go-to-definition** → use LSP / IDE navigation.
- **First step of repo exploration** → read README, directory tree, or entry points first.

### Anti-hallucination stop rule
If semantic results are low-relevance for the same task **2 times in a row**, **STOP**. Do not fabricate edits from weak matches. Ask user for a narrower starting point.

---

## 3. MCP TOOL REFERENCE

### Discovery
- `list_available_configs()` — list valid config names
- `list_available_indices(config)` — list indices for a config

### Search & lookup
- `semantic_search(config, query, index, top_k=5)` — embedding-based search
- `lookup_symbol(config, uid, index)` — check if a UID exists in the vector store

### Sidecar tools
- `inspect_sidecar(config, file_path, index)` — list short symbol names parsed from source
- `update_sidecar(config, file_path, index, symbol?, intent?, relations?)` — sync or update sidecar
- `clear_sidecar_symbol(config, file_path, symbol, index)` — remove a symbol from sidecar

### Ingestion & validation
- `refresh_knowledge(config, index, follow_symlinks=True)` — re-ingest files into vector store (**resource-intensive**)
- `validate_mesh(config)` — run integrity diagnostics

---

## 4. QUERY FORMULATION

SMAK uses **embedding-based semantic search**. Write queries as natural-language descriptions of behavior or purpose — not symbol names or file paths.

### Index selection guide

| What you need | Index |
|---|---|
| Code logic, design, implementation intent | `source_code` |
| Historical bug reports, known issues | `issues` |
| Test coverage, test cases | `tests` |
| Architecture docs, API docs | `documentation` |

### Good vs bad queries
```
Good: "CSV file row append logic"
Good: "error handling for out-of-range index in cell update"
Bad:  "append_row"         ← grep instead
Bad:  "csv_editor.py"      ← file tools instead
```

### Search result format
```json
{
  "hits": [{
    "uid": "/home/user/project/src/csv_editor.py::CsvEditor.update_cell",
    "exact_relative_path": "src/csv_editor.py",
    "match_type": "semantic",
    "score": 0.89,
    "content": "..."
  }],
  "related_context": [{
    "uid": "/home/user/project/issues/known-issues.md::*",
    "match_type": "relation",
    "source_hit": "/home/user/project/src/csv_editor.py::CsvEditor.update_cell",
    "content": "..."
  }]
}
```

**Field rules:**
- `uid` — full UID (`{absolute_path}::{symbol}`). Use in sidecar `relations` lists and with `lookup_symbol`.
- `exact_relative_path` — copy EXACTLY as `file_path` for sidecar tools. Never rewrite or guess.

---

## 5. SIDECAR OPERATIONS

### Sidecar YAML format
```yaml
# src/.csv_editor.py.sidecar.yaml
symbols:
  - name: CsvEditor                    # short symbol name (NOT full UID)
    intent: "Manages CSV read/write"
    relations:
      - "/home/user/project/issues/known-issues.md::*"    # full UID
      - "/home/user/project/src/other.py::OtherClass"     # full UID
  - name: CsvEditor.update_cell
    intent: ""
    relations: []
```

**Key distinction:**
- `name` field = **short symbol name** (e.g. `CsvEditor.update_cell`). Matches `inspect_sidecar` output.
- `relations` list = **full UIDs** (e.g. `/abs/path/file.py::Symbol`). Matches `semantic_search` hit UIDs.

### Workflow: inspect and update sidecar

```
# 1. Find the file via semantic search
semantic_search(config="my_config", query="CSV cell update logic", index="source_code")
# → hit: {"uid": "/home/.../src/csv_editor.py::CsvEditor.update_cell",
#          "exact_relative_path": "src/csv_editor.py", ...}

# 2. List short symbol names for the file
inspect_sidecar(config="my_config", file_path="src/csv_editor.py", index="source_code")
# → ["CsvEditor", "CsvEditor.append_row", "CsvEditor.update_cell", "CsvEditor.read_rows"]

# 3. Full sync (creates sidecar if missing, preserves existing metadata)
update_sidecar(config="my_config", file_path="src/csv_editor.py", index="source_code")
# → {"total_symbols": 4, "added": 4, "removed": 0, ...}

# 4. Update a specific symbol (use SHORT name from inspect_sidecar)
update_sidecar(
  config="my_config",
  file_path="src/csv_editor.py",
  index="source_code",
  symbol="CsvEditor.update_cell",           # ← short name
  intent="Rewrites entire file to update one cell.",
  relations=["/home/.../issues/known.md::*"] # ← full UIDs
)
```

### Workflow: clear a stale symbol

If full sync fails because a deleted symbol still has relations:
```
clear_sidecar_symbol(
  config="my_config",
  file_path="src/csv_editor.py",
  symbol="CsvEditor.old_method",     # ← short name
  index="source_code"
)
# Then re-run update_sidecar (full sync)
```

### `update_sidecar` modes

| Mode | When | What happens |
|---|---|---|
| Full sync (no `symbol`) | Create/sync sidecar | Creates if missing; preserves existing metadata; blocks removal of symbols with relations |
| Single-symbol (`symbol` given) | Update one symbol | Updates `intent` and/or `relations` for that symbol; at least one required |

---

## 6. COMMON PIPELINES

### Pipeline A: Add bi-directional sidecar relations between two folders

**Goal**: link code symbols in `src/` with issue entries in `issues/`.

**Pre-requisites**: both `source_code` and `issues` indices must be ingested (vectors exist in the store).

```
# Step 1 — Find the code symbol
semantic_search(config="cfg", query="CSV update logic", index="source_code")
# → code hit uid: "/home/.../src/csv_editor.py::CsvEditor.update_cell"

# Step 2 — Find the related issue
semantic_search(config="cfg", query="cell update out of range bug", index="issues")
# → issue hit uid: "/home/.../issues/csv-bugs.md::*"

# Step 3 — Verify both exist in their vector stores
lookup_symbol(config="cfg", uid="/home/.../src/csv_editor.py::CsvEditor.update_cell", index="source_code")
# → {"found": true, ...}
lookup_symbol(config="cfg", uid="/home/.../issues/csv-bugs.md::*", index="issues")
# → {"found": true, ...}

# Step 4 — Add relation: code → issue
inspect_sidecar(config="cfg", file_path="src/csv_editor.py", index="source_code")
update_sidecar(
  config="cfg", file_path="src/csv_editor.py", index="source_code",
  symbol="CsvEditor.update_cell",
  relations=["/home/.../issues/csv-bugs.md::*"]
)

# Step 5 — Add reverse relation: issue → code
inspect_sidecar(config="cfg", file_path="issues/csv-bugs.md", index="issues")
update_sidecar(
  config="cfg", file_path="issues/csv-bugs.md", index="issues",
  symbol="*",
  relations=["/home/.../src/csv_editor.py::CsvEditor.update_cell"]
)

# Step 6 — Validate mesh integrity
validate_mesh(config="cfg")
```

**Post-check**: run `semantic_search` on either index; `related_context` should now include the linked entity.

### Pipeline B: When and how to update the vector store

The vector store is **not** updated by sidecar operations. It must be explicitly refreshed.

**When to call `refresh_knowledge`:**
- After adding, modifying, or deleting source files
- After renaming or moving files
- When `semantic_search` returns stale/missing results
- When `lookup_symbol` returns `{"found": false}` for a file you know exists

**When NOT to call `refresh_knowledge`:**
- After updating sidecar metadata (intent/relations) — sidecars are read live at query time
- Routinely or "just in case" — it is resource-intensive

```
# Re-ingest a specific index
refresh_knowledge(config="cfg", index="source_code")
# → "Ingestion Complete! Processed Files: 42, Skipped Files: 3, Vectors Added: 210"

# Verify a specific symbol is now in the store
lookup_symbol(config="cfg", uid="/home/.../src/new_file.py::NewClass", index="source_code")
# → {"found": true, ...}
```

---

## 7. STRICT RULES

1. **`file_path` must exactly match `exact_relative_path`** from `semantic_search` hits. Never rewrite or guess.
2. **`symbol` parameter = short name** from `inspect_sidecar` (e.g. `CsvEditor.update_cell`). Never pass full UIDs as `symbol`.
3. **`relations` list = full UIDs** from `semantic_search` hits (e.g. `/abs/path::Symbol`).
4. **Always call `inspect_sidecar`** before `update_sidecar` (single-symbol mode) to confirm valid symbol names.
5. **Always call `lookup_symbol`** to verify a UID exists before adding it to `relations`.
6. **Sidecar updates ≠ vector store updates.** Changing a sidecar does not require re-ingestion.
7. **`refresh_knowledge` is resource-intensive.** Only call when source files have changed and you need updated search results.
