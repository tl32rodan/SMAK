# SMAK CLI Demo

This folder demonstrates a minimal end-to-end CLI flow for the passive SMAK kernel using a CSV editor sample.

## Run step by step

### 1) Ingest folders
```bash
cd demo
smak ingest --folder ./src --index source_code --config workspace_config.yaml --workers 1
smak ingest --folder ./issues --index issues --config workspace_config.yaml --workers 1
smak ingest --folder ./documentation --index documentation --config workspace_config.yaml --workers 1
```

### 2) Inspect canonical symbols
```bash
smak sidecar inspect ./src/csv_editor.py --config workspace_config.yaml
smak sidecar inspect ./src/tests/test_csv_editor.py --config workspace_config.yaml
smak sidecar inspect ./documentation/csv-editor-usage.md --config workspace_config.yaml
```

### 3) Create sidecar and add relations
```bash
smak sidecar init ./src/csv_editor.py --config workspace_config.yaml
cat > ./src/csv_editor.py.sidecar.yaml <<'YAML'
symbols:
  - name: CsvEditor
    intent: "Manage CSV rows for lightweight fixture editing"
    relations:
      - csv-editor-known-issues
  - name: CsvEditor.update_cell
    intent: "Update one cell by row/column index"
    relations:
      - csv-editor-known-issues
'YAML'
```

### 4) Re-ingest and query mesh context
```bash
smak ingest --folder ./src --index source_code --config workspace_config.yaml --workers 1
smak query "why update_cell fails" --index source_code --config workspace_config.yaml
```

You should see semantic hits from `csv_editor.py` and `related_context` entries that include `csv-editor-known-issues.md` content due to 1-hop relation traversal.

### 5) Run doctor
```bash
smak doctor --path . --index source_code --config workspace_config.yaml
```
