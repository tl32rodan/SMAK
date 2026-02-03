# SMAK

SMAK (Semantic Mesh Agentic Kernel) is a lightweight middleware layer that enforces
sidecar governance and mesh retrieval across source assets. Storage and orchestration
are delegated to Milvus Lite and LlamaIndex. It provides:

- **Ingestion pipeline**: Parse files into structured knowledge units, validate and
  enrich them with sidecar metadata, embed them, and persist vectors into Milvus Lite.
- **Semantic mesh tools**: Search across indices and resolve mesh relations through
  LlamaIndex-backed retrieval.
- **CLI workflow**: Initialize configuration, ingest folders, and run a Gradio-based
  agent server for chat interactions.

## Architecture overview

1. **Parsing**: Language-specific parsers (Python, Perl, Markdown issues, or
   line-based) convert file contents into `KnowledgeUnit` records.
2. **Sidecar validation**: `.sidecar.yaml`/`.sidecar.yml` files are validated in a
   fail-fast step before any ingestion continues.
3. **Metadata injection**: Each knowledge unit is enriched with sidecar fields such
   as `intent`, `mesh_relations`, and symbol metadata.
4. **Embedding**: Internal Nomic embeddings are generated for each unit, and the
   embedding dimension is auto-detected at runtime.
5. **Storage**: Embeddings and payloads are stored in Milvus Lite
   (`uri: ./milvus_data.db`).
6. **Agents**: Retrieval uses LlamaIndex ReAct with a Mesh Retrieval two-pass lookup
   to surface matches and follow mesh relations, served via a Gradio chat UI.

## Configuration

Use `smak init` to generate a template `workspace_config.yaml`. The schema includes:

```yaml
storage:
  provider: milvus_lite
  uri: ./milvus_data.db
llm:
  provider: qwen
  model: qwen3_235B_A22B
  temperature: 0.0
  # api_base: http://localhost:11434/v1
indices:
  - name: source_code
    description: Contains the project's source code (Python, Perl), function definitions, and logic.
  - name: issues
    description: Contains historical bug reports, GitHub issues, and Jira tickets describing known problems.
  - name: tests
    description: Contains unit tests, integration tests, and test cases.
  - name: documentation
    description: Contains architecture diagrams, API docs, and general knowledge base.
```

Use the Milvus Lite `uri` to define the local vector store location. Provide the
LLM provider details as needed for chat completion requests. SMAK auto-detects the
embedding dimension from the configured embedding service, so it does not need to
be listed in the config file.

## Ingestion workflow

```bash
smak ingest --folder ./src --index source_code --config workspace_config.yaml
```

During ingestion, SMAK:

1. Loads `workspace_config.yaml`.
2. Parses each file in the folder (Python, Perl, Markdown issues, or line-based).
3. Validates sidecar metadata and fails fast if invalid entries are found.
4. Injects sidecar metadata and mesh relations.
5. Embeds each knowledge unit with the internal Nomic model.
6. Stores vectors in Milvus Lite for the configured index.

You can adjust concurrency with `--workers` (default 4). If `tqdm` is installed
and stderr is a TTY, a progress bar is displayed during ingestion.

## Sidecar metadata

Sidecar files live alongside source files and use the suffix
`.sidecar.yaml` or `.sidecar.yml` (for example, `example.py.sidecar.yaml`).
They let you attach metadata and mesh relations to parsed symbols. Guidelines:

1. Keep sidecars next to the source file they annotate.
2. Ensure `symbol` values match the identifiers emitted by the parser.
3. Use `mesh_relations` to list related symbols for mesh traversal.
4. Store intent or ownership fields under `metadata`.

Keywords:
- **Mandatory**: `symbols` (list), `symbol` (per entry).
- **Optional but recommended**: `metadata` (per entry), `mesh_relations` (per entry).

Example sidecar:

```yaml
symbols:
  - symbol: SmakRepositoryLoader
    metadata:
      intent: "Ingest repository files into knowledge units"
      owner: "platform"
    mesh_relations:
      - SmakRepositoryLoader.parse
      - KnowledgeUnit
```

## Milvus Lite storage

SMAK uses a native Milvus Lite connection via `pymilvus` + `milvus-lite` to persist
and retrieve vector documents. Mesh relations are resolved by first retrieving
primary matches, then performing a second-pass lookup for related nodes referenced
in `mesh_relations` so agent tooling can traverse the semantic mesh. Install the
native dependencies with:

```bash
pip install pymilvus==2.6.8 milvus-lite==2.5.1
```

## LLM providers

SMAK supports internal OpenAI-compatible endpoints for Qwen and GPT-OSS models via
the `llm.provider` setting (`qwen` or `gpt-oss`). When using other providers, SMAK
falls back to the `llama_index.llms.openai_like` client using the configured model
and `api_base`.
