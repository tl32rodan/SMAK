---
name: smak-skill
description: SMAK (Semantic Mesh Augmented Kernel) - A semantic search and context expansion tool. Use this exclusively to explore code intent, historical context, and 1-hop bi-directional cross-entity relations (e.g., linking code with issues, tests, and docs in both directions).
---

# SMAK Skill (Compact / Strict)

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

### 2) MCP-facing API surface
`src/smak/mcp_server.py` exposes config-driven MCP tools:
- `list_available_configs()`
- `list_available_indices(config)`
- `refresh_knowledge(config, index="source_code", follow_symlinks=True)`
- `semantic_search(config, query, index="source_code", top_k=5)`
- `inspect_sidecar(config, file_path, index="source_code")`
- `init_sidecar(config, file_path, index="source_code")`
- `update_sidecar(config, file_path, updates, index="source_code")`
- `validate_mesh(config)`

> `registry.yaml` (or an equivalent registry file passed via `--registry`) is **mandatory**. The MCP server fails fast when the registry is missing or empty.

### 3) CLI utilities
- `smak init`
- `smak ingest`
- `smak query`
- `smak sidecar init|update|inspect`
- `smak doctor`

## QUERY & RELATION MODEL
`smak query` returns structured JSON with semantic and relational context separated:

```json
{
  "hits": [
    {"uid": "func_A", "exact_uid": "func_A", "exact_relative_path": "src/main.py", "match_type": "semantic", "score": 0.89, "content": "..."}
  ],
  "related_context": [
    {"uid": "issue_12", "match_type": "relation", "source_hit": "func_A", "content": "..."}
  ]
}
```

When calling sidecar APIs (`inspect_sidecar`, `init_sidecar`, `update_sidecar`), copy `hits[].exact_relative_path` directly as `file_path`. Do not rewrite or guess file paths.

### 1-Hop Semantic Mesh Traversal
1. Run vector search for `top_k` semantic hits.
2. Read `relations` metadata on those hits.
3. Fetch each related UID with `vector_store.get_by_id(uid)`.
4. Append those nodes under `related_context` (strictly one hop).

## WHEN NOT TO USE (Hard bans)
- DO NOT use SMAK for exact string matching (use `rg`, `grep`, or editor search).
- DO NOT use SMAK for go-to-definition or symbol jump (use LSP / IDE navigation).
- DO NOT use SMAK as the first step of repository exploration.

## WHEN TO USE (Triggers)
- **Intent discovery:** understand the "why" behind hacks, tradeoffs, or legacy behavior.
- **1-hop context expansion:** starting from a relevant code hit, fetch linked issues/docs/tests via relations.
- **Sidecar lifecycle operations:** when you need to inspect symbols or initialize/update intent+relations metadata for a concrete file selected from semantic search hits.

## STANDARD WORKFLOW (Must follow)
1. `list_available_configs` to pick a valid `config`.
2. If you are unsure which indices exist for that config, call `list_available_indices(config)` first and choose the exact target index.
3. `semantic_search` with natural-language intent query.
4. Choose one concrete hit and copy `exact_relative_path` exactly.
5. Use sidecar tools in order as needed:
   - `inspect_sidecar` to verify symbol UIDs available for that file.
   - `init_sidecar` to scaffold sidecar entries.
   - `update_sidecar` to write `intent` / `relations` updates.

### sidecar parameter rule (Strict)
- `file_path MUST EXACTLY match exact_relative_path from semantic_search hits`.
- Never rewrite, normalize, or guess the path manually.

## ANTI-HALLUCINATION STOP RULE
- If semantic results are low-relevance for the same task **2 times in a row**, STOP.
- Do not fabricate edits from weak matches.
- Ask user for a narrower starting point (file, module, symbol, or concrete symptom).
