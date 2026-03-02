# SMAK

SMAK now stands for **Semantic Mesh Augmented Kernel** and is focused as a **passive MCP knowledge kernel**.

It is the source-of-truth layer for:
- code symbols,
- sidecar intent metadata,
- issue/document relations,
- and semantic retrieval context.

## What SMAK provides

### 1) Ingestion kernel
SMAK ingests files into knowledge units, enriches them with sidecar metadata, computes embeddings, and stores vectors.

### 2) MCP-facing API surface
`src/smak/mcp_server.py` exposes strict multi-workspace MCP tools:
- `list_available_workspaces()`
- `refresh_knowledge(workspace, folder=".", index="source_code", follow_symlinks=True)`
- `semantic_search(workspace, query, index="source_code", top_k=5)`
- `manage_sidecar(workspace, action, file_path, updates=None, reingest=False, index="source_code")`
- `validate_mesh(workspace, path=".")`

> `all_workspaces.yaml` (or an equivalent registry file passed via `--registry`) is now **mandatory**. The MCP server fails fast when the registry is missing or empty.

### 3) CLI utilities
- `smak init`
- `smak ingest`
- `smak query`
- `smak sidecar init|update|inspect`
- `smak doctor`

> Deprecated commands removed: `search`, `stats`.

---

## Query JSON output

`smak query` returns structured JSON with semantic and relational context separated:

```json
{
  "hits": [
    {"uid": "func_A", "match_type": "semantic", "score": 0.89, "content": "..."}
  ],
  "related_context": [
    {"uid": "issue_12", "match_type": "relation", "source_hit": "func_A", "content": "..."}
  ]
}
```

### 1-Hop Semantic Mesh Traversal
1. Run vector search for `top_k` semantic hits.
2. Read `relations` metadata on those hits.
3. Fetch each related UID with `vector_store.get_by_id(uid)`.
4. Append those nodes under `related_context` (strictly one hop).

---

## Quick start (CLI)

```bash
smak ingest --index source_code --config demo/workspace_a/workspace_config.yaml
smak ingest --index issues      --config demo/workspace_a/workspace_config.yaml
smak query "why does update_cell raise an error" \
      --index source_code \
      --config demo/workspace_a/workspace_config.yaml
smak doctor --config demo/workspace_a/workspace_config.yaml
```

## Quick start (MCP server)

```bash
python -m smak.mcp_server --registry ./demo/all_workspaces.yaml
```

The registry file must define workspace names and filesystem paths. All tool calls except
`list_available_workspaces` require explicitly choosing one workspace.

See **[`demo/README.md`](demo/README.md)** for a complete step-by-step walkthrough of both
workspaces, covering every CLI command and MCP tool call with expected output.
