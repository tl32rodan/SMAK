# SMAK

For SMAK concepts, MCP workflow conventions, and agent behavior guidance, see
**[`smak-skill/SKILL.md`](smak-skill/SKILL.md)**.

## Quick start (CLI)

```bash
smak ingest --index source_code --config demo/workspace_a/workspace_config.yaml
smak ingest --index issues      --config demo/workspace_a/workspace_config.yaml
smak search "why does update_cell raise an error" \
      --index source_code \
      --config demo/workspace_a/workspace_config.yaml
smak doctor --config demo/workspace_a/workspace_config.yaml
```

## Quick start (MCP server)

```bash
python -m smak.mcp_server
```

The MCP server starts stateless — no config required at startup.
Every tool call accepts a `config` parameter (path to `workspace_config.yaml`)
so the agent can dynamically select which workspace to operate on.

```bash
# Optional: provide a custom embedding setup
python -m smak.mcp_server --embedding-setup ./custom_embedding.yaml
```

See **[`demo/README.md`](demo/README.md)** for a complete step-by-step walkthrough of both
workspaces, covering every CLI command and MCP tool call with expected output.
