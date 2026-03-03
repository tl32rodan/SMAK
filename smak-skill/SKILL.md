---
name: smak-skill
description: SMAK (Semantic Mesh Augmented Kernel) - A semantic search and context expansion tool. Use this exclusively to explore code intent, historical context, and 1-hop bi-directional cross-entity relations (e.g., linking code with issues, tests, and docs in both directions).
---

# SMAK Skill (Compact / Strict)

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
2. `semantic_search` with natural-language intent query.
3. Choose one concrete hit and copy `exact_relative_path` exactly.
4. Use sidecar tools in order as needed:
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
