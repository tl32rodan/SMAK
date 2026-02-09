# SMAK CLI Demo

This folder demonstrates a minimal end-to-end CLI flow for the passive SMAK kernel using a CSV editor sample.

## Included structure
- `src/` source code folder
  - `csv_editor.py` demo code
  - `tests/test_csv_editor.py` behavioral unit tests
- `issues/` issue notes and known limitations
- `documentation/` usage notes
- `workspace_config.yaml` sample config

## Run step by step

### 1) Ingest all three folders first
```bash
cd demo
smak ingest --folder ./src --index source_code --config workspace_config.yaml --workers 1
smak ingest --folder ./issues --index issues --config workspace_config.yaml --workers 1
smak ingest --folder ./documentation --index documentation --config workspace_config.yaml --workers 1
```

### 2) Fetch symbols from several file paths
```bash
smak search ./src/csv_editor.py --config workspace_config.yaml
smak search ./src/tests/test_csv_editor.py --config workspace_config.yaml
smak search ./documentation/csv-editor-usage.md --config workspace_config.yaml
```

### 3) Create sidecar and add content
```bash
smak sidecar init ./src/csv_editor.py --config workspace_config.yaml
cat > ./src/csv_editor.py.sidecar.yaml <<'YAML'
symbols:
  - name: CsvEditor
    intent: "Manage CSV rows for lightweight fixture editing"
    relations:
      - ISSUE-001
  - name: CsvEditor.update_cell
    intent: "Update one cell by row/column index"
    relations:
      - ISSUE-002
YAML
```

### 4) Run doctor
```bash
smak doctor --path .
```

## Expected artifacts
- `src/csv_editor.py.sidecar.yaml` created and enriched
- doctor returns `Mesh diagnostics passed.`
- ingest reports processed files and vectors added
