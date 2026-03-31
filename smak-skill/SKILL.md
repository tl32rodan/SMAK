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
| **Vector store** | Stores embeddings of every ingested code unit. Updated by `refresh_knowledge` (ingestion). Queried by `semantic_search`. |
| **Sidecar file** | Hidden YAML (e.g. `src/.foo.py.sidecar.yaml`) storing `intent` and `relations` per symbol. Updated by `update_sidecar`. Read at query time for 1-hop expansion. |
| **UID** | Globally unique identifier for a vector-store entry: `{path}::{symbol}` (e.g. `/home/user/project/src/foo.py::ClassName.method` or `$DDI_ROOT_PATH/src/foo.py::ClassName.method`). |
| **Symbol name** | Short name without path prefix (e.g. `ClassName.method`). Used in sidecar `name` fields and as the `symbol` parameter in sidecar tools. |
| **path_env** | Optional config field mapping UIDs to environment variables instead of absolute paths (e.g. `$DDI_ROOT_PATH/...`). |

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

### Config structure

```
workspace_config.yaml              ← agent passes path dynamically per tool call
  └── indices:
        - name: source_code   paths: [./src]   path_env: DDI_ROOT_PATH
        - name: issues         paths: [./issues]
        - name: tests          paths: [./src/tests]
        - name: documentation  paths: [./documentation]
```

- Every SMAK tool takes `config` as its first parameter — the path to `workspace_config.yaml`.
- `list_available_indices(config)` → valid `index` values.
- Indices are **not limited to the 4 defaults** — you can define any number with any names.

---

## 2. WHEN TO USE / NOT USE

### MANDATORY PRE-CHECK — before every `semantic_search` call

**Do you already know the file path or function/class name?**
- If **YES** → **DO NOT** call `semantic_search`. Use `grep`/`rg` for the symbol name, or read the file directly. Semantic search is **only** for when you do NOT know where something lives.
- If **NO** → proceed with `semantic_search`.

**Violation examples** (never do these):
```
# BAD: You already know the function name is "append_row"
semantic_search(query="append_row function", ...)

# BAD: You already know the file is src/csv_editor.py
semantic_search(query="csv editor module", ...)
```

### Use SMAK for
- **Intent discovery**: understand the "why" behind hacks, tradeoffs, or legacy behavior — when you do NOT already know where the relevant code is.
- **1-hop context expansion**: from a code hit, auto-fetch linked issues/docs/tests via relations.
- **Sidecar lifecycle**: inspect, create, or update `intent` + `relations` metadata for source files.
- **Exploratory search**: finding code related to a concept, behavior, or purpose when you have no specific file/symbol in mind.
- **Cross-index discovery**: use `multi_index_search` to find related entities across code, issues, tests, and docs simultaneously.

### Do NOT use SMAK for
- **Exact string matching** → use `rg` / `grep`.
- **Go-to-definition** → use LSP / IDE navigation.
- **First step of repo exploration** → read README, directory tree, or entry points first.
- **When you already know the file path or symbol name** → read the file directly or use `grep`.
- **Re-finding something you already found** → if a previous tool call already returned the file/symbol location, use that result directly.

### Anti-hallucination stop rule
If semantic results are low-relevance for the same task **2 times in a row**, **STOP**. Do not fabricate edits from weak matches. Ask user for a narrower starting point.

---

## 3. MCP TOOL REFERENCE

All tools take `config` (path to `workspace_config.yaml`) as their first parameter.

### Discovery
- `list_available_indices(config)` — list indices for a workspace

### Search & lookup
- `semantic_search(config, query, index, top_k=5)` — embedding-based search
- `multi_index_search(config, query, indices=None, top_k=3)` — search across ALL (or specified) indices at once
- `lookup_symbol(config, uid, index)` — check if a UID exists in the vector store

### Sidecar tools
- `inspect_sidecar(config, file_path, index)` — list short symbol names parsed from source
- `update_sidecar(config, file_path, index, symbol?, intent?, relations?)` — sync or update sidecar
- `clear_sidecar_symbol(config, file_path, symbol, index)` — remove a symbol from sidecar
- `batch_update_sidecars(config, file_paths, index)` — sync sidecars for multiple files at once

### Ingestion & validation
- `refresh_knowledge(config, index, follow_symlinks=True)` — re-ingest files into vector store (**resource-intensive**)
- `validate_mesh(config)` — run integrity diagnostics
- `workspace_status(config)` — per-index stats dashboard (vector count, last update, etc.)

---

## 4. QUERY FORMULATION

**Prerequisite**: you have already confirmed that you do NOT know the file path or symbol name (see Section 2 pre-check). If you know either, stop — do not formulate a query. Use `grep` or read the file directly.

SMAK uses **embedding-based semantic search**. Write queries as natural-language descriptions of behavior or purpose — not symbol names or file paths.

### Index selection guide

Use `list_available_indices()` first to see what indices exist. Indices are **project-specific** — there is no fixed set. Common patterns:

| What you need | Typical index name |
|---|---|
| Code logic, design, implementation intent | `source_code`, `rtl_code`, `verification` |
| Historical bug reports, known issues | `issues`, `jira_tickets` |
| Test coverage, test cases | `tests`, `regression_tests` |
| Architecture docs, API docs | `documentation`, `specs` |

When unsure which index to use, use `multi_index_search` to search across all indices at once.

### Good vs bad queries
```
Good: "CSV file row append logic"
Good: "error handling for out-of-range index in cell update"
Good: "retry logic for network failures"
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
- `uid` — full UID (`{path}::{symbol}`). Use in sidecar `relations` lists and with `lookup_symbol`.
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
      - "$DDI_ROOT_PATH/src/other.py::OtherClass"         # env var UID
  - name: CsvEditor.update_cell
    intent: ""
    relations: []
```

**Key distinction:**
- `name` field = **short symbol name** (e.g. `CsvEditor.update_cell`). Matches `inspect_sidecar` output.
- `relations` list = **full UIDs** (e.g. `/abs/path/file.py::Symbol` or `$ENV_VAR/path/file.py::Symbol`). Matches `semantic_search` hit UIDs.

### Workflow: inspect and update sidecar

```
# 1. Find the file via semantic search
semantic_search(config="./workspace_config.yaml", query="CSV cell update logic", index="source_code")
# → hit: {"uid": "...", "exact_relative_path": "src/csv_editor.py", ...}

# 2. List short symbol names for the file
inspect_sidecar(config="./workspace_config.yaml", file_path="src/csv_editor.py", index="source_code")
# → ["CsvEditor", "CsvEditor.append_row", "CsvEditor.update_cell", "CsvEditor.read_rows"]

# 3. Full sync (creates sidecar if missing, preserves existing metadata)
update_sidecar(config="./workspace_config.yaml", file_path="src/csv_editor.py", index="source_code")

# 4. Update a specific symbol (use SHORT name from inspect_sidecar)
update_sidecar(
  config="./workspace_config.yaml",
  file_path="src/csv_editor.py",
  index="source_code",
  symbol="CsvEditor.update_cell",
  intent="Rewrites entire file to update one cell.",
  relations=["/home/.../issues/known.md::*"]
)
```

### Workflow: batch sidecar initialization

```
# Initialize sidecars for all files at once
batch_update_sidecars(config="./workspace_config.yaml", file_paths=["src/a.py", "src/b.py", "src/c.py"], index="source_code")
```

### Workflow: clear a stale symbol

If full sync fails because a deleted symbol still has relations:
```
clear_sidecar_symbol(
  config="./workspace_config.yaml",
  file_path="src/csv_editor.py",
  symbol="CsvEditor.old_method",
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

```
cfg = "./workspace_config.yaml"

# Step 1 — Find the code symbol
semantic_search(config=cfg, query="CSV update logic", index="source_code")

# Step 2 — Find the related issue
semantic_search(config=cfg, query="cell update out of range bug", index="issues")

# Step 3 — Verify both exist in their vector stores
lookup_symbol(config=cfg, uid="...", index="source_code")
lookup_symbol(config=cfg, uid="...", index="issues")

# Step 4 — Add relation: code → issue
update_sidecar(config=cfg, file_path="src/csv_editor.py", index="source_code",
  symbol="CsvEditor.update_cell",
  relations=["...issues/csv-bugs.md::*"])

# Step 5 — Add reverse relation: issue → code
update_sidecar(config=cfg, file_path="issues/csv-bugs.md", index="issues",
  symbol="*",
  relations=["...src/csv_editor.py::CsvEditor.update_cell"])

# Step 6 — Validate mesh integrity
validate_mesh(config=cfg)
```

### Pipeline B: Multi-index exploration

Use `multi_index_search` for broad discovery across all knowledge:

```
# Search all indices at once
multi_index_search(config="./workspace_config.yaml", query="authentication timeout handling", top_k=3)
# → returns results grouped by index: {"source_code": {...}, "issues": {...}, ...}
```

### Pipeline C: Workspace health check

```
cfg = "./workspace_config.yaml"

# Quick overview of all indices
workspace_status(config=cfg)
# → {"indices": [{"name": "source_code", "vector_count": 42, "last_update": "..."}]}

# Detailed integrity check
validate_mesh(config=cfg)
```

---

## 7. INDEX DESIGN PATTERNS

### Indices are arbitrary — not limited to 4 defaults

The default `smak init` template creates 4 indices (`source_code`, `issues`, `tests`, `documentation`), but you can define **any number** with **any names**. Examples:

```yaml
# EDA / semiconductor project
indices:
  - name: rtl_code
    description: "Verilog/SystemVerilog RTL modules for DDR5 PHY datapath"
    paths: [$DDI_ROOT_PATH/rtl/phy]
    path_env: DDI_ROOT_PATH
  - name: verification
    description: "UVM testbenches and coverage models for PHY verification"
    paths: [$DDI_ROOT_PATH/verif]
    path_env: DDI_ROOT_PATH
  - name: constraints
    description: "SDC timing constraints and floorplan definitions"
    paths: [$DDI_ROOT_PATH/constraints]
  - name: release_notes
    description: "Release notes, known issues, and ECO history"
    paths: [./release_notes]
```

### Guidelines for splitting vs. merging indices

- **Split** when content types have fundamentally different search intents (code vs. issues vs. docs).
- **Merge** when two directories share the same search intent and you want cross-results.
- Each index has its own vector store — smaller indices search faster.
- Relations can cross index boundaries, so splitting doesn't prevent linking.

---

## 8. WRITING EFFECTIVE DESCRIPTIONS

The `description` field is the **agent's ONLY hint** for index selection. Write it precisely.

### Good descriptions
```yaml
description: "RTL Verilog modules for DDR5 PHY datapath, including FIFO, serializer, and DQ/DQS logic"
description: "UVM testbenches and coverage models for PHY verification"
description: "Historical Jira tickets and postmortem reports for timing closure failures"
```

### Bad descriptions
```yaml
description: "source code"          # too vague — what kind? what domain?
description: "files"                # meaningless
description: "tests"                # which tests? for what?
```

### Description checklist
- Include **file types** present (Verilog, Python, Perl, Markdown)
- Include **domain terms** (DDR5, PHY, datapath, authentication, etc.)
- Describe **what questions** this index answers
- Be **specific enough** to disambiguate from other indices

---

## 9. ENVIRONMENT VARIABLE UIDs

### When to use `path_env`

Use `path_env` when your codebase lives at different absolute paths depending on the context:
- Different release versions at different root directories
- SOS/version-control workspaces vs. production ("online") paths
- Shared NFS mounts with user-specific prefixes

### How it works

```yaml
# workspace_config.yaml
indices:
  - name: source_code
    description: "Production source code"
    paths: [$DDI_ROOT_PATH/src]
    path_env: DDI_ROOT_PATH
```

When `path_env` is set:
- UIDs are stored as `$DDI_ROOT_PATH/src/a.py::ClassName` instead of `/opt/ddi/online/src/a.py::ClassName`
- At query time, `$DDI_ROOT_PATH` is expanded to the current environment value
- Sidecar relations can reference `$DDI_ROOT_PATH/...` UIDs

### Path mismatch warnings

When you edit sidecar files in an SOS workspace (different path than the env var root), SMAK emits a **warning** — this is expected:

```
WARNING: Path mismatch: sidecar at '/workspace/user1/src/a.py' has relation
targeting '$DDI_ROOT_PATH/issues/bug.md::*' (env root: '/opt/ddi/online').
This is expected when editing in an SOS workspace.
```

---

## 10. STRICT RULES

1. **Never use `semantic_search` when you already know the target.** If you know the file path, function name, or class name — use `grep`/`rg` or read the file directly.
2. **`file_path` must exactly match `exact_relative_path`** from `semantic_search` hits. Never rewrite or guess.
3. **`symbol` parameter = short name** from `inspect_sidecar` (e.g. `CsvEditor.update_cell`). Never pass full UIDs as `symbol`.
4. **`relations` list = full UIDs** from `semantic_search` hits (e.g. `/abs/path::Symbol` or `$ENV_VAR/path::Symbol`).
5. **Always call `inspect_sidecar`** before `update_sidecar` (single-symbol mode) to confirm valid symbol names.
6. **Always call `lookup_symbol`** to verify a UID exists before adding it to `relations`.
7. **Sidecar updates ≠ vector store updates.** Changing a sidecar does not require re-ingestion.
8. **`refresh_knowledge` is resource-intensive.** Only call when source files have changed and you need updated search results.
