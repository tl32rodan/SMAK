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
PYTHONPATH=../src python -c "from smak.cli import main; main()" ingest --folder ./src --index source_code --config workspace_config.yaml --workers 1
PYTHONPATH=../src python -c "from smak.cli import main; main()" ingest --folder ./issues --index issues --config workspace_config.yaml --workers 1
PYTHONPATH=../src python -c "from smak.cli import main; main()" ingest --folder ./documentation --index documentation --config workspace_config.yaml --workers 1
```

### 2) Fetch symbols from several file paths
```bash
PYTHONPATH=../src python -c "from smak.cli import main; main()" search ./src/csv_editor.py --config workspace_config.yaml
PYTHONPATH=../src python -c "from smak.cli import main; main()" search ./src/tests/test_csv_editor.py --config workspace_config.yaml
PYTHONPATH=../src python -c "from smak.cli import main; main()" search ./documentation/csv-editor-usage.md --config workspace_config.yaml
```

### 3) Create sidecar and add content
```bash
PYTHONPATH=../src python -c "from smak.cli import main; main()" sidecar init ./src/csv_editor.py --config workspace_config.yaml
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
PYTHONPATH=../src python -c "from smak.cli import main; main()" doctor --path .
```

## Expected artifacts
- `src/csv_editor.py.sidecar.yaml` created and enriched
- doctor returns `Mesh diagnostics passed.`
- ingest reports processed files and vectors added
