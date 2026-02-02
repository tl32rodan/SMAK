# SMAK

SMAK (Semantic Mesh Agentic Kernel) is a lightweight middleware layer that enforces
sidecar governance and mesh retrieval across source assets. Storage and orchestration
are delegated to Milvus Lite and LlamaIndex. It provides:

- **Ingestion pipeline**: Parse files into structured knowledge units, validate and
  enrich them with sidecar metadata, embed them, and persist vectors into Milvus Lite.
- **Semantic mesh tools**: Search across indices and resolve mesh relations through
  LlamaIndex-backed retrieval.
- **CLI workflow**: Initialize configuration, ingest folders, and run a demo server
  for agent interactions.

## Architecture overview

1. **Parsing**: `SmakRepositoryLoader` uses language-specific parsers (Python, Perl,
   Markdown issues, or line-based) to convert file contents into `KnowledgeUnit` records.
2. **Sidecar validation**: `.sidecar.yaml`/`.sidecar.yml` files are validated in a
   fail-fast step before any ingestion continues.
3. **Metadata injection**: Each knowledge unit is enriched with `file_name`,
   `symbol_name`, `intent`, and `mesh_relations` metadata.
4. **Embedding**: Internal Nomic models generate embeddings for each unit.
5. **Storage**: Embeddings and payloads are stored in Milvus Lite
   (`uri: ./milvus_data.db`).
6. **Agents**: Retrieval uses LlamaIndex ReAct with a Mesh Retrieval two-pass lookup
   to surface matches and follow mesh relations.

## Configuration

Use `smak init` to generate a template `workspace_config.yaml`. The schema includes:

```yaml
storage:
  provider: milvus_lite
  uri: ./milvus_data.db
llm:
  provider: openai
  model: llama3
  temperature: 0.2
  api_base: http://localhost:11434/v1
indices:
  - name: source_code
    description: Source code files
embedding_dimensions: 3
```

Use the Milvus Lite `uri` to define the local vector store location. Provide the
LLM provider details as needed for chat completion requests.

## Ingestion workflow

```bash
smak ingest --folder ./src --index source_code --config workspace_config.yaml
```

During ingestion, SMAK:

1. Loads `workspace_config.yaml`.
2. `SmakRepositoryLoader` parses each file in the folder.
3. Validates sidecar metadata and fails fast if invalid entries are found.
4. Injects `file_name`, `symbol_name`, `intent`, and `mesh_relations`.
5. Embeds each knowledge unit with the internal Nomic model.
6. Stores vectors in Milvus Lite for the configured index.

## LlamaIndex + Milvus Lite storage

SMAK uses the LlamaIndex vector store integration with Milvus Lite to persist and
retrieve vector documents. Mesh relations are resolved by first retrieving primary
matches, then performing a second-pass lookup for related nodes referenced in
`mesh_relations` so agent tooling can traverse the semantic mesh.
