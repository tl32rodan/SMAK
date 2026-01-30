# SMAK

SMAK (Semantic Mesh Agentic Kernel) is a lightweight prototype for ingesting source
assets into vector storage and querying them through agent tools. It provides:

- **Ingestion pipeline**: Parse files into structured knowledge units, enrich them
  with sidecar metadata, embed them, and persist vectors into an index.
- **Semantic mesh tools**: Search across indices and expand relations between
  related entities.
- **CLI workflow**: Initialize configuration, ingest folders, and run a demo server
  for agent interactions.

## Architecture overview

1. **Parsing**: Language-specific parsers (Python, Perl, Markdown issues, or line-based)
   convert file contents into `KnowledgeUnit` records.
2. **Sidecar metadata**: Optional `.sidecar.yaml`/`.sidecar.yml` files provide
   structured metadata and relations keyed by symbols.
3. **Embedding**: The embedder turns each knowledge unit into vector embeddings.
4. **Storage**: The vector adapter persists embeddings and payloads into a registry-backed
   vector index (for example, `faiss-storage-lib`).
5. **Agents**: Tools can query indices and follow relations in the semantic mesh.

## Configuration

Use `smak init` to generate a template `workspace_config.yaml`. The schema includes:

```yaml
indices:
  - name: source_code
    description: Source code files
llm:
  provider: openai
  model: llama3
storage:
  base_path: vault
embedding_dimensions: 3
```

The CLI uses the `storage.base_path` when creating the vector index registry.

## Ingestion workflow

```bash
smak ingest --folder ./src --index source_code --config workspace_config.yaml
```

During ingestion, SMAK:

1. Loads `workspace_config.yaml`.
2. Creates a registry instance (see protocol below).
3. Parses each file in the folder.
4. Applies sidecar metadata if a `file.ext.sidecar.yaml` exists.
5. Embeds each knowledge unit.
6. Adds vector documents to the specified index.

## Registry protocol for vector storage libraries

SMAK relies on an index registry abstraction so you can plug in backends such as
`faiss-storage-lib`. The CLI defaults to:

```python
from faiss_storage_lib.engine.registry import IndexRegistry
registry = IndexRegistry(base_path)
```

For alternative libraries, provide an object that implements the following protocol:

```python
class IndexRegistry(Protocol):
    def get_index(self, name: str) -> VectorIndex: ...

class VectorIndex(Protocol):
    def add(self, docs: Sequence[VectorDocument]) -> None: ...
```

`VectorDocument` is a structure containing a unique ID, a vector, and a payload
with metadata and relations. For agent search operations, the registry should also
expose search-capable indices:

```python
class VectorSearchIndex(Protocol):
    def search(self, query: str) -> Sequence[dict[str, Any]]: ...
    def get_by_id(self, uid: str) -> dict[str, Any] | None: ...
```

To wire in a custom registry for libraries like `faiss-storage-lib`, keep the same
`get_index` contract and ensure the returned index implements `add`, `search`, and
`get_by_id` as needed by the ingestion pipeline and agent tools.
