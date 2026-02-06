# SMAK CLI Demo

This folder demonstrates a minimal end-to-end CLI flow for the passive SMAK kernel.

## Included files
- `src/auth.py` sample source
- `workspace_config.yaml` sample config

## Run

```bash
cd demo
PYTHONPATH=../src python -m smak.cli search ./src/auth.py --config workspace_config.yaml
PYTHONPATH=../src python -m smak.cli sidecar init ./src/auth.py --config workspace_config.yaml
PYTHONPATH=../src python -m smak.cli doctor --path .
PYTHONPATH=../src python -m smak.cli ingest --folder ./src --index source_code --config workspace_config.yaml --workers 1
```

## Expected artifacts
- `src/auth.py.sidecar.yaml` created by `sidecar init`
- doctor returns `Mesh diagnostics passed.`
- ingest reports processed files and vectors added
