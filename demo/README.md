# SMAK Strict Multi-Workspace Demo

This demo now models the **mandatory registry-based multi-workspace architecture**.

## Layout

- `all_workspaces.yaml`: workspace registry used by MCP routing.
- `workspace_a/`: primary demo flow containing source code, issues, and docs.
- `workspace_b/`: isolated secondary workspace with its own `workspace_config.yaml`.

## Registry

`all_workspaces.yaml` maps logical workspace names to filesystem paths:

- `demo_flow_a -> ./workspace_a`
- `demo_flow_b -> ./workspace_b`

Use this file with the MCP server:

```bash
python -m smak.mcp_server --registry ./demo/all_workspaces.yaml
```

## CLI walkthrough in workspace A

```bash
cd demo/workspace_a
smak ingest --folder ./src --index source_code --config workspace_config.yaml --workers 1
smak ingest --folder ./issues --index issues --config workspace_config.yaml --workers 1
smak ingest --folder ./documentation --index documentation --config workspace_config.yaml --workers 1
```

Inspect symbols:

```bash
smak sidecar inspect ./src/csv_editor.py --config workspace_config.yaml
smak sidecar inspect ./src/tests/test_csv_editor.py --config workspace_config.yaml
smak sidecar inspect ./documentation/csv-editor-usage.md --config workspace_config.yaml
smak sidecar inspect ./issues/csv-editor-known-issues.md --config workspace_config.yaml
```

Re-ingest and query:

```bash
smak ingest --folder ./src --index source_code --config workspace_config.yaml --workers 1
smak query "why update_cell fails" --index source_code --config workspace_config.yaml
smak doctor --path . --index source_code --config workspace_config.yaml
```

## Isolation intent

`workspace_b` intentionally contains separate files/config so you can validate that tool calls routed to `demo_flow_b` do not read `workspace_a` content.
