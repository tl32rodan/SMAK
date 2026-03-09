---
name: smak-skill
description: SMAK (Semantic Mesh Augmented Kernel) - A semantic search and context expansion tool. Use this exclusively to explore code intent, historical context, and 1-hop bi-directional cross-entity relations (e.g., linking code with issues, tests, and docs in both directions).
---

# SMAK Skill (Compact / Strict)

## QUICK START (4 calls, common case)

> Goal: find code that handles a specific concern, then write intent metadata for it.

**Step 1 — Discover valid configs**
```
list_available_configs()
# → {"demo_flow_a": {"config_path": "..."}, "demo_flow_b": {...}}
```

**Step 2 — Discover indices for your chosen config**
```
list_available_indices(config="demo_flow_a")
# → [{"name": "source_code", ...}, {"name": "issues", ...}, ...]
```

**Step 3 — Search semantically**
```
semantic_search(
    config="demo_flow_a",
    query="how cell updates are written back to the CSV file",
    index="source_code",
    top_k=5
)
```
Copy `hits[0].exact_relative_path` verbatim from the result. Do not retype it.

**Step 4 — Sync sidecar and write intent/relations per symbol**
```
# Full sync (creates sidecar if missing, preserves existing relations)
update_sidecar(
    config="demo_flow_a",
    file_path="src/csv_editor.py",   ← exact_relative_path from step 3
    index="source_code"
)

# Then update individual symbols
update_sidecar(
    config="demo_flow_a",
    file_path="src/csv_editor.py",
    index="source_code",
    symbol="/home/user/project/src/csv_editor.py::CsvEditor.update_cell",
    intent="Rewrites entire file to update one cell. Raises IndexError on out-of-range row/col.",
    relations=["/home/user/project/issues/csv-editor-known-issues.md::*"]
)
```

---

## WHAT SMAK IS (Authoritative)
SMAK stands for **Semantic Mesh Augmented Kernel** and acts as a **passive MCP knowledge kernel**.

It is the source-of-truth layer for:
- code symbols,
- sidecar intent metadata,
- issue/document relations,
- semantic retrieval context.

## CORE CAPABILITIES

### 1) Ingestion kernel
SMAK ingests files into knowledge units, enriches them with sidecar metadata, computes embeddings, and stores vectors.

### 2) Config hierarchy
Every SMAK operation requires two levels of config:

```
registry.yaml                          ← pass --registry to MCP server
  └── configs:
        <config_name>:
          config_path: ./workspace/workspace_config.yaml   ← SmakConfig
            └── indices:
                  - name: source_code   path: ./src
                  - name: issues        path: ./issues
                  - name: tests         path: ./src/tests
                  - name: documentation path: ./documentation
```

- Use `list_available_configs()` to find valid `config` values.
- Use `list_available_indices(config)` to find valid `index` values for a config.

### 3) MCP-facing API surface
`src/smak/mcp_server.py` exposes config-driven MCP tools:
- `list_available_configs()`
- `list_available_indices(config)`
- `refresh_knowledge(config, index="source_code", follow_symlinks=True)`
- `semantic_search(config, query, index="source_code", top_k=5)`
- `inspect_sidecar(config, file_path, index="source_code")`
- `update_sidecar(config, file_path, index="source_code", symbol=None, intent=None, relations=None)`
- `clear_sidecar_symbol(config, file_path, symbol, index="source_code")`
- `validate_mesh(config)`

> `registry.yaml` (or an equivalent registry file passed via `--registry`) is **mandatory**. The MCP server fails fast when the registry is missing or empty.

### 4) CLI utilities
- `smak init`
- `smak ingest`
- `smak query`
- `smak sidecar update|clear|inspect`
- `smak doctor`

---

## QUERY FORMULATION (Critical)

SMAK uses **embedding-based semantic search**. Queries match on meaning and intent, not on literal text. Write queries as natural-language descriptions of behavior or purpose.

### Index selection guide

| What you need | Index to use |
|---|---|
| Understand code logic, design, or implementation intent | `source_code` |
| Find historical bug reports, known issues, tickets | `issues` |
| Find test coverage, test cases, test scenarios | `tests` |
| Find architecture docs, API docs, knowledge base entries | `documentation` |

### Good queries (intent/behavior-based)
```
"CSV file row append logic"
"error handling for out-of-range index in cell update"
"how log entries are parsed from structured log files"
"authentication flow when token expires"
"retry logic for network failures"
```

### Bad queries (do not use these patterns with SMAK)
```
"append_row"         ← exact function name → use grep/ripgrep instead
"csv_editor.py"      ← exact filename → use file tools instead
"def parse"          ← code syntax → use grep instead
"TODO fix bug"       ← too vague; try the issues index with a symptom description
```

---

## QUERY & RELATION MODEL

`semantic_search` returns structured JSON with semantic hits and relational context separated:

```json
{
  "hits": [
    {
      "uid": "/home/user/project/src/csv_editor.py::CsvEditor.update_cell",
      "exact_relative_path": "src/csv_editor.py",
      "match_type": "semantic",
      "score": 0.89,
      "content": "..."
    }
  ],
  "related_context": [
    {
      "uid": "/home/user/project/issues/csv-editor-known-issues.md::*",
      "match_type": "relation",
      "source_hit": "/home/user/project/src/csv_editor.py::CsvEditor.update_cell",
      "content": "..."
    }
  ]
}
```

**Field reference:**
- `uid` — globally unique identifier in format `{absolute_path}::{symbol}`; use this value in sidecar `relations` lists
- `exact_relative_path` — copy this EXACTLY as `file_path` when calling sidecar tools; never rewrite or guess
- `score` — similarity score returned by the underlying vector store; do not rely on absolute values or thresholds — the direction and scale vary by backend implementation
- `match_type: "semantic"` — retrieved directly by vector search
- `match_type: "relation"` — auto-fetched via 1-hop traversal from a sidecar relation

### 1-Hop Semantic Mesh Traversal
1. Run vector search for `top_k` semantic hits.
2. Read `relations` metadata on those hits (from sidecar YAML files).
3. Fetch each related UID with `vector_store.get_by_id(uid)`.
4. Append those nodes under `related_context` (strictly one hop).

---

## SIDECAR WORKFLOW (Step-by-Step)

Use this workflow when you need to read or update sidecar intent/relation metadata for a source file.

### Step 1 — Semantic search to find the target file
```
semantic_search(
  config="demo_flow_a",
  query="CSV file row append logic",
  index="source_code",
  top_k=3
)
```
→ Returns hit: `{"uid": "/home/user/project/src/csv_editor.py::CsvEditor.append_row", "exact_relative_path": "src/csv_editor.py", "score": 0.89, ...}`

### Step 2 — Inspect available symbol UIDs
```
inspect_sidecar(
  config="demo_flow_a",
  file_path="src/csv_editor.py",
  index="source_code"
)
```
→ Returns: `["/home/user/project/src/csv_editor.py::CsvEditor", "/home/user/project/src/csv_editor.py::CsvEditor.append_row", "/home/user/project/src/csv_editor.py::CsvEditor.update_cell", "/home/user/project/src/csv_editor.py::CsvEditor.read_rows"]`

> Always call `inspect_sidecar` before `update_sidecar` (single-symbol mode) to confirm valid symbol UIDs.

### Step 3 — Sync sidecar with current source symbols
```
update_sidecar(
  config="demo_flow_a",
  file_path="src/csv_editor.py",
  index="source_code"
)
```
→ Creates sidecar if it doesn't exist, or syncs it to match the current symbols in the source file.
→ Existing `intent` and `relations` are **preserved** for symbols that still exist.
→ New symbols get empty stubs.
→ Returns: `{"file_path": "...", "sidecar_path": "...", "total_symbols": 4, "added": 0, "removed": 0}`

> If sync is blocked because a deleted symbol still has relations, the error message lists the exact `clear_sidecar_symbol` calls needed.

### Step 4 — Update intent and relations per symbol
```
update_sidecar(
  config="demo_flow_a",
  file_path="src/csv_editor.py",
  index="source_code",
  symbol="/home/user/project/src/csv_editor.py::CsvEditor",
  intent="Manages read, append, and in-place cell-update operations on CSV files.",
  relations=["/home/user/project/issues/csv-editor-known-issues.md::*"]
)

update_sidecar(
  config="demo_flow_a",
  file_path="src/csv_editor.py",
  index="source_code",
  symbol="/home/user/project/src/csv_editor.py::CsvEditor.update_cell",
  intent="Rewrites the entire file to update one cell at the given row/column position. Raises IndexError when the row or column index is out of range (see ISSUE-001).",
  relations=["/home/user/project/issues/csv-editor-known-issues.md::*"]
)
```
→ Returns: `{"file_path": "...", "sidecar_path": "...", "total_symbols": 4}`

**`update_sidecar` modes:**

| Mode | When | What happens |
|---|---|---|
| Full sync (no `symbol`) | Create/sync sidecar | Creates if missing; preserves existing metadata; blocks removal of symbols with relations |
| Single-symbol (`symbol` provided) | Update one symbol | Updates only `intent` and/or `relations` for that symbol; at least one must be given |

### Step 5 — Clear a symbol (if needed before sync)

If sync fails because a removed symbol still has relations, clear it first:
```
clear_sidecar_symbol(
  config="demo_flow_a",
  file_path="src/csv_editor.py",
  symbol="/home/user/project/src/csv_editor.py::CsvEditor.old_method",
  index="source_code"
)
```
→ Removes the symbol entry from the sidecar regardless of its relations.
→ Then re-run `update_sidecar` (full sync) to complete the cleanup.

---

## SIDECAR YAML SCHEMA

Sidecar files are stored on disk as hidden YAML files next to the source file.

**Naming convention:**
- Source file `src/csv_editor.py` → sidecar `src/.csv_editor.py.sidecar.yaml`

**Format:**
```yaml
symbols:
  - name: /home/user/project/src/csv_editor.py::ClassName       # symbol UID — must match inspect_sidecar output
    intent: "What this symbol does"                              # free-text; empty string if unknown
    relations:                                                   # list of related UIDs (other symbols, issues, docs)
      - "/home/user/project/issues/csv-editor-known-issues.md::*"   # UID of an entity in the issues index
      - "/home/user/project/src/other.py::OtherClass.some_method"   # UID of another code symbol
  - name: /home/user/project/src/csv_editor.py::ClassName.method_name
    intent: ""
    relations: []
```

**How relations are resolved at query time:**
When `semantic_search` returns a hit, SMAK automatically loads the sidecar for that file and fetches every related UID via `vector_store.get_by_id(uid)`. These appear in `related_context` with `match_type: "relation"`.

---

## WHEN NOT TO USE (Hard bans)
- DO NOT use SMAK for exact string matching — use `rg`, `grep`, or editor search.
  - Wrong: `semantic_search(query="IndexError")` to find all places that raise IndexError.
  - Right: `rg "IndexError" src/`
- DO NOT use SMAK for go-to-definition or symbol jump — use LSP / IDE navigation.
  - Wrong: `semantic_search(query="CsvEditor.update_cell")` to locate the method definition.
  - Right: LSP go-to-definition on the symbol.
- DO NOT use SMAK as the first step of repository exploration — read the README, directory tree, or entry points first to orient yourself before running semantic search.

## WHEN TO USE (Triggers)
- **Intent discovery:** understand the "why" behind hacks, tradeoffs, or legacy behavior.
- **1-hop context expansion:** starting from a relevant code hit, fetch linked issues/docs/tests via relations.
- **Sidecar lifecycle operations:** when you need to inspect symbols or initialize/update intent+relations metadata for a concrete file selected from semantic search hits.

---

## STANDARD WORKFLOW (Must follow)
1. `list_available_configs` to pick a valid `config`.
2. If you are unsure which indices exist for that config, call `list_available_indices(config)` first and choose the exact target index (see **Index selection guide** above).
3. `semantic_search` with a natural-language intent query (see **Query Formulation** above).
4. Choose one concrete hit and copy `exact_relative_path` exactly.
5. Use sidecar tools in order as needed (see **Sidecar Workflow** above):
   - `inspect_sidecar` to verify symbol UIDs available for that file.
   - `update_sidecar` (no `symbol`) to sync the sidecar with current source symbols.
   - `update_sidecar` (with `symbol`) to write `intent` / `relations` for individual symbols.
   - `clear_sidecar_symbol` to remove a symbol entry before re-syncing (only when sync is blocked).

### sidecar parameter rule (Strict)
- `file_path MUST EXACTLY match exact_relative_path from semantic_search hits`.
- Never rewrite, normalize, or guess the path manually.

---

## ANTI-HALLUCINATION STOP RULE
- If semantic results are low-relevance for the same task **2 times in a row**, STOP.
- Do not fabricate edits from weak matches.
- Ask user for a narrower starting point (file, module, symbol, or concrete symptom).
