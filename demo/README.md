# SMAK CLI Demo

This folder demonstrates a minimal end-to-end CLI flow for the passive SMAK kernel using a CSV editor sample.

## Included structure
- `src/` source code folder
  - `csv_editor.py` demo code
  - `tests/test_csv_editor.py` behavioral unit tests
- `issues/` issue notes and known limitations
- `documentation/` usage notes
- `workspace_config.yaml` sample config

## Run

```bash
cd demo
PYTHONPATH=../src python -m smak.cli search ./src/csv_editor.py --config workspace_config.yaml
PYTHONPATH=../src python -m smak.cli sidecar init ./src/csv_editor.py --config workspace_config.yaml
PYTHONPATH=../src python -m smak.cli doctor --path .
PYTHONPATH=../src python -m unittest demo.src.tests.test_csv_editor
PYTHONPATH=../src python -m smak.cli ingest --folder ./src --index source_code --config workspace_config.yaml --workers 1
```

## Expected artifacts
- `src/csv_editor.py.sidecar.yaml` created by `sidecar init`
- doctor returns `Mesh diagnostics passed.`
- unit tests pass for CSV editor behavior
- ingest reports processed files and vectors added
