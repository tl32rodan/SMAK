---
name: sos-smak-skill
description: CliosoftSOS environment guide for SMAK. Teaches agents the internal EDA version control workflow — online vs. version control vs. SOS workspace — and how to correctly use SMAK (path_env, sidecar editing, ingestion) within this environment. Load this skill when working in projects that use CliosoftSOS and $DDI_ROOT_PATH.
---

# CliosoftSOS + SMAK Workflow

## 1. WHAT CLIOSOFT SOS IS

CliosoftSOS is the **version control system** used internally for EDA (Electronic Design Automation) projects. It is NOT git. Key differences:

| Concept | Git | CliosoftSOS |
|---|---|---|
| Repository | `.git/` directory | Centralized SOS server |
| Branch | git branch | "flow release" (版控) |
| Main | `main` / `master` | "online" |
| Working copy | `git clone` → local dir | SOS workspace (link-based snapshot) |
| Commit | `git commit` + `git push` | `sos check-in` |
| Checkout | `git checkout` | `sos check-out` into workspace |

SOS workspaces are **link snapshots**: the SOS server creates a directory structure made of symlinks pointing to the actual files on a shared disk. When you check out a file, SOS replaces the link with a real writable copy. When you check in, it goes back to the shared location.

## 2. THE THREE-LAYER PATH MODEL

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Layer 1: Online (production / main)                             │
│    $DDI_ROOT_PATH = /CAD/stdcell                                 │
│    Access:  READ-ONLY (shared disk, visible to everyone)         │
│    Purpose: The "truth" — production codebase                    │
│    FAISS:   Primary index is built from this layer               │
│                                                                  │
│  Layer 2: Version Control (flow releases / 版控)                  │
│    $DDI_ROOT_PATH = /CAD/stdcell_production/{version_string}/    │
│    Access:  READ-ONLY (shared disk, visible to everyone)         │
│    Purpose: Frozen snapshots of specific releases                │
│    FAISS:   Each version can have its own FAISS index            │
│    Note:    Different DDI_ROOT_PATH value per version            │
│                                                                  │
│  Layer 3: SOS Workspace (personal checkout)                      │
│    Path:    /arbitrary/path/created/by/sos/                      │
│    Access:  READ-WRITE (the only place you can edit files)       │
│    Purpose: Personal working area for making changes             │
│    Created: By calling SOS server to create a link snapshot      │
│    Target:  Can target online (Layer 1) or any version (Layer 2) │
│    Sidecar: All sidecar edits happen here                        │
│    Check-in: Pushes changes back to Layer 1 or Layer 2           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Path examples

```
# Online (Layer 1)
/CAD/stdcell/rtl/phy/dq_serdes.v
/CAD/stdcell/verif/tb_phy_top.sv
/CAD/stdcell/doc/releases/eco_042.md

# Version control (Layer 2) — e.g., version "2024Q3_RC1"
/CAD/stdcell_production/2024Q3_RC1/rtl/phy/dq_serdes.v
/CAD/stdcell_production/2024Q3_RC1/verif/tb_phy_top.sv

# SOS Workspace (Layer 3) — user "john", workspace "ws_fix_timing"
/users/john/ws_fix_timing/rtl/phy/dq_serdes.v      ← link or real file
/users/john/ws_fix_timing/verif/tb_phy_top.sv       ← link
```

### Key rules

- **Online and version control are read-only.** You cannot edit files there directly.
- **All edits happen in an SOS workspace.** After editing, `sos check-in` pushes changes back.
- **`$DDI_ROOT_PATH` is the abstraction** that points to either online or a specific version.
- **The workspace path is NOT `$DDI_ROOT_PATH`.** Workspace is a temporary working area at an unrelated path.

## 3. HOW $DDI_ROOT_PATH MAPS TO SMAK

SMAK's `path_env` feature bridges the gap between SOS's multi-path model and SMAK's UID system.

### Config setup

```yaml
# workspace_config.yaml
indices:
  - name: rtl_code
    description: "Verilog/SystemVerilog RTL modules for DDR5 PHY datapath, including DQ/DQS serializers, FIFO, and clock domain crossing logic"
    paths:
      - $DDI_ROOT_PATH/rtl/phy
    path_env: DDI_ROOT_PATH

  - name: verification
    description: "UVM testbenches, coverage models, and assertion libraries for PHY functional verification"
    paths:
      - $DDI_ROOT_PATH/verif
    path_env: DDI_ROOT_PATH

  - name: constraints
    description: "SDC timing constraints, floorplan DEF, and power intent UPF for PHY implementation"
    paths:
      - $DDI_ROOT_PATH/constraints
    path_env: DDI_ROOT_PATH

  - name: release_notes
    description: "ECO history, release notes, known issues, and waiver documentation"
    paths:
      - $DDI_ROOT_PATH/doc/releases
    path_env: DDI_ROOT_PATH
```

### What happens at each layer

| Action | Layer | $DDI_ROOT_PATH | What SMAK does |
|---|---|---|---|
| `ingest` | Online or Version Control | `/CAD/stdcell` or `/CAD/stdcell_production/v1/` | Builds FAISS index with UIDs like `$DDI_ROOT_PATH/rtl/phy/dq_serdes.v::dq_serializer` |
| `search` | Any (reads FAISS) | Expanded at runtime | Queries the FAISS index, resolves `$DDI_ROOT_PATH` to find sidecar files |
| `enrich_symbol` | Workspace | `/CAD/stdcell` (or version) | Writes sidecar in workspace; relations use `$DDI_ROOT_PATH` (not workspace path) |
| `check_health` | Any | Expanded at runtime | Validates sidecar relations resolve correctly |

### UID format in SOS environment

```
$DDI_ROOT_PATH/rtl/phy/dq_serdes.v::dq_serializer

When DDI_ROOT_PATH=/CAD/stdcell, expands to:
/CAD/stdcell/rtl/phy/dq_serdes.v::dq_serializer

When DDI_ROOT_PATH=/CAD/stdcell_production/2024Q3_RC1, expands to:
/CAD/stdcell_production/2024Q3_RC1/rtl/phy/dq_serdes.v::dq_serializer
```

## 4. WORKFLOW: ENRICH SIDECAR IN A WORKSPACE

You are in an SOS workspace at `/users/john/ws_fix_timing/`. The FAISS index was built from online (`DDI_ROOT_PATH=/CAD/stdcell`).

```python
cfg = "/users/john/ws_fix_timing/workspace_config.yaml"

# 1. Search the FAISS index (built from online)
hit = search(config=cfg, query="DQ serializer timing-critical path", index="rtl_code")
# → uid: "$DDI_ROOT_PATH/rtl/phy/dq_serdes.v::dq_serializer"
# → exact_relative_path: "rtl/phy/dq_serdes.v"

# 2. Find the related ECO document
eco = search(config=cfg, query="timing closure ECO for DQ path", index="release_notes")
# → uid: "$DDI_ROOT_PATH/doc/releases/eco_042.md::*"

# 3. Verify the ECO UID exists
lookup(config=cfg, uid="$DDI_ROOT_PATH/doc/releases/eco_042.md::*", index="release_notes")
# → {"found": true}

# 4. Annotate the RTL symbol
enrich_symbol(
  config=cfg,
  file_path="rtl/phy/dq_serdes.v",
  symbol="dq_serializer",
  intent="8:1 serializer for DQ lane. Timing-critical — see ECO-042 for hold fix.",
  relations=["$DDI_ROOT_PATH/doc/releases/eco_042.md::*"],
  index="rtl_code"
)
# → WARNING: Path mismatch (expected — you're in a workspace)
# → Sidecar written to: /users/john/ws_fix_timing/rtl/phy/.dq_serdes.v.sidecar.yaml
# → Relations use $DDI_ROOT_PATH, NOT /users/john/ws_fix_timing/

# 5. After SOS check-in:
#    Sidecar lands at: /CAD/stdcell/rtl/phy/.dq_serdes.v.sidecar.yaml
#    Relations point to $DDI_ROOT_PATH paths — correct ✓
```

## 5. PATH MISMATCH WARNING — WHEN IT'S OK

When editing sidecars in a workspace, SMAK emits:

```
WARNING: Path mismatch: sidecar at '/users/john/ws_fix_timing/rtl/phy/dq_serdes.v'
has relation targeting '$DDI_ROOT_PATH/doc/releases/eco_042.md::*'
(env root: '/CAD/stdcell'). This is expected when editing in an SOS workspace.
```

**This is normal.** It means:
- You're editing in a workspace (Layer 3)
- Relations correctly point to the canonical path (Layer 1/2)
- After `sos check-in`, the sidecar will be at the canonical path and everything aligns

**When it's NOT ok:**
- You see this warning but you're NOT in an SOS workspace
- `$DDI_ROOT_PATH` is set to the wrong value (e.g., pointing to wrong version)

## 6. WORKFLOW: INGEST AFTER VERSION CONTROL RELEASE

When a new version control release is cut:

```bash
# New release at /CAD/stdcell_production/2024Q3_RC1/
export DDI_ROOT_PATH=/CAD/stdcell_production/2024Q3_RC1
```

```python
cfg = "/CAD/stdcell_production/2024Q3_RC1/workspace_config.yaml"

# Rebuild FAISS for the new version
ingest(config=cfg, index="rtl_code")
ingest(config=cfg, index="verification")
ingest(config=cfg, index="release_notes")

# Verify
check_health(config=cfg)
# → May report stale relations from previous version — expected
```

UIDs in the new FAISS index will resolve to `/CAD/stdcell_production/2024Q3_RC1/...` paths.

### Maintaining online after a release

Online (`/CAD/stdcell`) continues to evolve independently. Its FAISS index should be periodically re-ingested:

```bash
export DDI_ROOT_PATH=/CAD/stdcell
```

```python
cfg = "/CAD/stdcell/workspace_config.yaml"
ingest(config=cfg, index="rtl_code")
```

## 7. WHICH LAYER TO TARGET

| Task | Target Layer | $DDI_ROOT_PATH |
|---|---|---|
| Build FAISS for production | Online | `/CAD/stdcell` |
| Build FAISS for a release | Version Control | `/CAD/stdcell_production/{version}/` |
| Edit sidecars | Workspace (Layer 3) | Set to the target layer (1 or 2) |
| Search/query | Any (reads FAISS) | Set to match the FAISS you want to query |
| Run `check_health` | Match the FAISS layer | Same as ingest |

## 8. STRICT RULES FOR SOS ENVIRONMENTS

1. **Never hardcode absolute paths in relations.** Always use `$DDI_ROOT_PATH/...` format.
2. **Never edit sidecars directly in online or version control.** Always use an SOS workspace.
3. **Always set `path_env: DDI_ROOT_PATH`** in config for indices on the shared disk.
4. **Set `$DDI_ROOT_PATH` before running SMAK** — it determines which layer you're operating on.
5. **Path mismatch warnings in workspaces are normal.** Don't suppress or work around them.
6. **Re-ingest after cutting a version control release** to build FAISS for the new version.
7. **One FAISS index per `$DDI_ROOT_PATH` value.** Don't share indices across online and version control.
8. **`workspace_config.yaml` can live anywhere** — in the workspace, in online, or separate. The `paths` field with `$DDI_ROOT_PATH` determines what gets indexed, not where the config lives.
