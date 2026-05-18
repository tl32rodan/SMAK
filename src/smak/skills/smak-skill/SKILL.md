---
name: smak-skill
description: SMAK (Semantic Mesh Augmented Kernel) - A semantic search and context expansion tool exposed via both a CLI (`smak ...`) and an MCP server. Use ONLY when you do NOT already know the file path or symbol name. If you know the target, use grep or read the file directly. Use this exclusively to explore code intent, historical context, and 1-hop bi-directional cross-entity relations (e.g., linking code with issues, tests, and docs in both directions).
---

# SMAK Skill

## 1. WHAT SMAK IS

SMAK (**Semantic Mesh Augmented Kernel**) is a **passive knowledge kernel** — a read/write layer for:
- **Semantic retrieval**: embedding-based search over code, issues, tests, and docs.
- **Sidecar metadata**: per-file YAML storing human-written `intent` and cross-entity `relations` for each code symbol.
- **1-hop mesh traversal**: when a search hit has sidecar relations, SMAK auto-fetches the linked entities and returns them alongside the hit.

### Two interfaces — same underlying operations

SMAK exposes every operation through **two equivalent surfaces**. Pick whichever fits the caller:

| Surface | Invocation | When to use |
|---|---|---|
| **CLI** | `smak <command> [--json] ...` | Shell scripts, one-off commands, agents that shell out (e.g. All-Might), CI |
| **MCP** | Tool call on a running `python -m smak.mcp_server` | Long-lived agent sessions that already speak MCP |

Both delegate to the same `core_ops` layer, so behavior is identical. Pass `--json` to the CLI to get the same structured payload an MCP tool would return.

### Key concepts

| Concept | Description |
|---|---|
| **Vector store** | Embeddings of every ingested code unit. Updated by `ingest`. Queried by `search`. |
| **Sidecar file** | Hidden YAML (e.g. `src/.foo.py.sidecar.yaml`) storing `intent` and `relations` per symbol. Updated by `enrich_symbol` / `enrich_file`. Read at query time for 1-hop expansion. |
| **UID** | Globally unique identifier: `{path}::{symbol}` (e.g. `/home/user/project/src/foo.py::ClassName.method` or `$DDI_ROOT_PATH/src/foo.py::ClassName.method`). |
| **Symbol name** | Short name without path prefix (e.g. `ClassName.method`). Used in `enrich_symbol`. |
| **`env` block** | Top-level section in `workspace_config.yaml` defining workspace-scoped variables. All `$VAR` references in `paths` and `uri` resolve from this block — never from shell environment. |
| **`$SMAK_DATA`** | Convention: variable pointing to vector store directory. Defined in `env:`. Used in `uri` fields. |
| **`$DDI_ROOT_PATH`** | Convention (EDA/SOS): variable pointing to the project root. Defined in `env:`. Used in `paths`. |

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
env:
  DDI_ROOT_PATH: /opt/ddi/online
  SMAK_DATA: /data/smak

indices:
  - name: source_code
    description: "RTL Verilog modules for DDR5 PHY datapath"
    paths: [$DDI_ROOT_PATH/src]
    uri: $SMAK_DATA/source_code
  - name: issues
    description: "Jira tickets and postmortems for timing closure failures"
    paths: [./issues]
    uri: $SMAK_DATA/issues
```

Every SMAK operation needs a workspace config — passed as the first parameter to an MCP
tool (`config=...`) or via `--config <path>` on the CLI. Indices are **not limited to any
default set** — define any number with any names.

**`env` block** defines workspace-scoped variables. All `$VAR` references in `paths` and `uri`
resolve from this block — **never from the shell environment**. This makes configs fully
self-contained and portable.

**`uri` must be absolute or use a `$VAR`** from the `env` block. Relative paths are rejected.

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

## 3. TOOL REFERENCE (CLI + MCP)

Every operation requires a workspace config. The MCP tool takes `config` as the first
argument; the CLI takes it as `--config <path>` (defaults to `./workspace_config.yaml`).
All CLI commands also accept `--json` to emit the same structured payload the MCP tool returns.

### Discovery

| Purpose | MCP tool | CLI command |
|---|---|---|
| List all indices (call this first) | `describe_workspace(config)` | `smak describe --config <cfg>` |

### Search

| Purpose | MCP tool | CLI command |
|---|---|---|
| Semantic search within one index | `search(config, query, index, top_k=5)` | `smak search "<query>" --index <n> --top-k 5 --config <cfg>` |
| Search across all (or specified) indices | `search_all(config, query, indices=None, top_k=3)` | `smak search-all "<query>" [--indices <n>]... --top-k 3 --config <cfg>` |
| Verify a UID exists in the vector store | `lookup(config, uid, index)` | `smak lookup "<uid>" --index <n> --config <cfg>` |

### Sidecar enrichment

| Purpose | MCP tool | CLI command |
|---|---|---|
| Annotate one symbol (auto-syncs sidecar, validates symbol, clears stale entries) | `enrich_symbol(config, file_path, symbol, intent?, relations?, index, bidirectional=False, dry_run=False)` | `smak enrich --file <p> --symbol <s> [--intent <t>] [--relation <uid>]... [--bidirectional] [--dry-run] --index <n> --config <cfg>` |
| Create/sync a file's sidecar (stubs for all symbols) | `enrich_file(config, file_path, index)` | `smak enrich-file --file <p> --index <n> --config <cfg>` |
| Sync sidecars for multiple files at once | `enrich_batch(config, file_paths, index)` | `smak enrich-batch <p1> <p2> ... --index <n> --config <cfg>` |

When `--bidirectional` is set, the reverse relation is also written back from each target.
When `--dry-run` is set, the enriched sidecar is computed and returned without being written
(useful in SOS environments — see `sos-smak-skill/SKILL.md`).

### Maintenance

| Purpose | MCP tool | CLI command |
|---|---|---|
| Re-embed files into a vector store (**resource-intensive**) | `ingest(config, index, follow_symlinks=True)` | `smak ingest --index <n> [--no-follow-symlinks] --config <cfg>` |
| Integrity diagnostics, returns `{status, issues}` | `check_health(config)` | `smak health --config <cfg>` |
| Knowledge graph coverage stats (per-index totals, relations, coverage %, asymmetric-relation warnings) | `graph_stats(config)` | `smak stats --config <cfg>` |

### CLI-only commands

These two commands exist on the CLI but have no MCP counterpart (they are bootstrapping /
end-user concerns, not agent operations):

| Purpose | CLI command |
|---|---|
| Write a starter `workspace_config.yaml` to disk | `smak init [--path <p>] [--force]` |
| Run mesh diagnostics with a non-zero exit code on failure (script-friendly variant of `health`) | `smak doctor --config <cfg>` |

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

MCP:
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

CLI:
```bash
smak enrich \
  --config ./workspace_config.yaml \
  --index source_code \
  --file src/csv_editor.py \
  --symbol CsvEditor.update_cell \
  --intent "Rewrites entire file to update one cell. Known IndexError issue." \
  --relation '$DDI_ROOT_PATH/issues/csv-bugs.md::*' \
  --json
```

What `enrich_symbol` / `smak enrich` does internally:
1. Validates `symbol` exists in the file (returns error + valid list if not)
2. Syncs the sidecar (creates if missing, auto-clears stale symbols)
3. Writes intent and relations

### Initialize sidecars for a directory

MCP:
```python
# Create stub sidecars for all files
enrich_batch(
  config="./workspace_config.yaml",
  file_paths=["src/a.py", "src/b.py", "src/c.py"],
  index="source_code"
)
```

CLI:
```bash
smak enrich-batch src/a.py src/b.py src/c.py \
  --index source_code \
  --config ./workspace_config.yaml \
  --json
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

MCP:
```python
cfg = "./workspace_config.yaml"

# 1. Find the code
hit = search(config=cfg, query="CSV cell update logic", index="source_code")

# 2. Find the related issue
issue = search(config=cfg, query="cell update out of range bug", index="issues")

# 3. Verify the issue UID exists
lookup(config=cfg, uid=issue_uid, index="issues")

# 4. Annotate the code symbol with the relation (use bidirectional=True to also write the reverse)
enrich_symbol(
  config=cfg,
  file_path=hit["exact_relative_path"],
  symbol="CsvEditor.update_cell",
  relations=[issue_uid],
  index="source_code",
  bidirectional=True,
)
```

CLI:
```bash
CFG=./workspace_config.yaml

# 1. Find the code
smak search "CSV cell update logic" --index source_code --config "$CFG" --json

# 2. Find the related issue
smak search "cell update out of range bug" --index issues --config "$CFG" --json

# 3. Verify the issue UID exists
smak lookup "$ISSUE_UID" --index issues --config "$CFG" --json

# 4. Annotate the code symbol (--bidirectional adds the reverse relation in one step)
smak enrich \
  --config "$CFG" --index source_code \
  --file src/csv_editor.py \
  --symbol CsvEditor.update_cell \
  --relation "$ISSUE_UID" \
  --bidirectional \
  --json
```

### Pipeline B: Broad exploration

```python
# MCP — don't know which index? Search everything.
search_all(config=cfg, query="authentication timeout handling", top_k=3)
```

```bash
# CLI equivalent
smak search-all "authentication timeout handling" --top-k 3 --config "$CFG" --json
```

### Pipeline C: Health check

```python
# MCP
check_health(config=cfg)
# → {"status": "healthy", "issues": []}
# → {"status": "unhealthy", "issues": ["Orphaned sidecar: ..."]}
```

```bash
# CLI — same structured payload with --json; or use `smak doctor` to exit non-zero on failure.
smak health --config "$CFG" --json
smak doctor --config "$CFG"   # for scripts: exits 1 if unhealthy
```

---

## 7. INDEX DESIGN PATTERNS

Indices are **arbitrary** — not limited to any default set.

```yaml
# EDA project example
env:
  DDI_ROOT_PATH: /opt/ddi/online
  SMAK_DATA: /data/smak

indices:
  - name: rtl_code
    description: "Verilog/SystemVerilog RTL modules for DDR5 PHY datapath"
    paths: [$DDI_ROOT_PATH/rtl/phy]
    uri: $SMAK_DATA/rtl_code
  - name: verification
    description: "UVM testbenches and coverage models"
    paths: [$DDI_ROOT_PATH/verif]
    uri: $SMAK_DATA/verification
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

## 8. WORKSPACE-SCOPED VARIABLES (`env` block)

Variables are defined in the `env:` block of `workspace_config.yaml`. They are **not**
shell environment variables — SMAK never reads `os.environ`.

```yaml
env:
  DDI_ROOT_PATH: /opt/ddi/online
  SMAK_DATA: /data/smak

indices:
  - name: source_code
    paths: [$DDI_ROOT_PATH/src]
    uri: $SMAK_DATA/source_code
```

| Convention | Purpose |
|---|---|
| `SMAK_DATA` | Root directory for vector store data (used in `uri`) |
| `DDI_ROOT_PATH` | Project/codebase root (used in `paths`; common in EDA/SOS) |

- **`uri`** uses `$SMAK_DATA` — keeps vector store location portable.
- **`paths`** uses `$DDI_ROOT_PATH` — lets the same config work across workspaces.
- During ingest, UIDs are collapsed to `$DDI_ROOT_PATH/src/a.py::ClassName` automatically
  (longest matching env value wins).
- Any undefined `$VAR` reference raises an error at config load time.

For CliosoftSOS / EDA environments, see **[`sos-smak-skill/SKILL.md`](../sos-smak-skill/SKILL.md)** for the full three-layer path model.

---

## 9. STRICT RULES

1. **Never use `search` when you already know the target.**
2. **`file_path` must exactly match `exact_relative_path`** from search hits.
3. **`symbol` = short name** (e.g. `CsvEditor.update_cell`), not full UID.
4. **`relations` = full UIDs** (e.g. `/abs/path::Symbol` or `$ENV_VAR/path::Symbol`).
5. **Always `lookup`** to verify a UID exists before adding it to relations.
6. **Sidecar updates ≠ vector store updates.** Call `ingest` only when source files change.
7. **`ingest` is resource-intensive.** Don't call casually.
8. **`uri` must be absolute or use `$VAR` from the `env` block.** Relative URIs are rejected.
9. **Variables resolve from `env:` only.** SMAK never reads shell environment variables.
