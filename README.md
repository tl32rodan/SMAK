# SMAK

For SMAK concepts, MCP workflow conventions, and agent behavior guidance, see
**[`smak-skill/SKILL.md`](smak-skill/SKILL.md)**.

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
python -m smak.mcp_server --registry ./demo/registry.yaml
```

The registry file must list config file paths. All tool calls except
`list_available_configs` require explicitly choosing one config.

See **[`demo/README.md`](demo/README.md)** for a complete step-by-step walkthrough of both
workspaces, covering every CLI command and MCP tool call with expected output.
