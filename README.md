# Persistence Landscapes for Topological Analog Retrieval in Market Risk Forecasting

**Ismail Guzel**

This repository contains Python code, a frozen market-data panel, analysis
outputs, and figure-generation scripts for studying topological analog
retrieval in one-day-ahead market-risk forecasting. The method represents
historical return windows with persistence landscapes and uses similar windows
as analogs for forecasting extreme portfolio losses.

## Quick start

Use Python 3.9.6 and install the pinned dependencies:

```bash
python3 -m pip install -r requirements.txt
```

A conda environment with the same pins is provided in
[`environment/environment.yml`](environment/environment.yml):

```bash
conda env create -f environment/environment.yml
conda activate tda-analog-risk
```

The exact environment used to produce the shipped outputs is recorded in
[`environment/revision_pip_freeze.txt`](environment/revision_pip_freeze.txt).

## Verify the shipped results

```bash
./scripts/reproduce_final.sh --verify-existing
```

This check takes a few seconds and runs no new analysis. It verifies the
frozen panel (hash, row count, date range), the configuration registries, the
schemas and radius isolation of the final outputs, the declared null-control
and k-sensitivity designs, and the SHA-256 hash of every frozen input and
result listed in
[`provenance/result_hash_manifest.sha256`](provenance/result_hash_manifest.sha256).

## Generate figures

The two figures are stylized synthetic examples and need no analysis outputs:

```bash
python3 scripts/mainfigures_pipeline_concept.py
python3 scripts/mainfigures_dtw_topology.py
```

The scripts write `fig1_pipeline_concept.pdf`, `fig2_mechanism_dtw_topology.pdf`,
and `table_mechanism_dtw_topology.csv` to `figures/`.

## Rerun the analysis

```bash
./scripts/reproduce_final.sh --plan        # print the execution plan; computes nothing
./scripts/reproduce_final.sh --build-only  # rebuild the window-length table and the figures
./scripts/reproduce_final.sh --full        # rerun the complete analysis
```

`--full` runs, in order: blocked validation over the prespecified candidate
set, the locked evaluation of 13 finalist families at exclusion radii 0, 125,
and 250, the repeated history-only null controls (500 draws per control and
radius), the validation k-sensitivity design, paired proper-loss inference and
radius-specific model confidence sets, the exclusion and cutoff-tie
diagnostics, and the two post hoc mechanism diagnostics. It overwrites
`results/` and takes several hours. It ends with a schema and design check;
the byte-level hash check is skipped there because recomputed outputs record
fresh run times.

## Tests

```bash
python3 -m pytest
```

## Project layout

| Path | Contents |
|---|---|
| `tda_risk/` | Topology features, analog retrieval, null controls, forecasting, scoring, and inference code |
| `scripts/` | Analysis, verification, and figure-generation entry points |
| `data/` | Frozen aligned market-data panel and metadata |
| `revision_config/` | Analysis protocol, candidate and finalist registries, random-seed registry |
| `results/` | Shipped analysis outputs |
| `figures/` | Generated figures |
| `provenance/` | SHA-256 manifests of the frozen inputs, outputs, and package files |
| `environment/` | Environment specification and the recorded package versions |
| `tests/` | Automated tests |

## Data

The analysis uses `data/panel_published.csv.gz` and does not download or
refresh external data. Its date splits and forecast conventions are recorded
in [`revision_config/revision_protocol.json`](revision_config/revision_protocol.json).
The evaluation period is a fixed reanalysis panel, not a pristine untouched
holdout. See [`data/README.md`](data/README.md) for data sources and
[`results/README.md`](results/README.md) for a description of every output.
