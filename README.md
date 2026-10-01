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

## Generate figures

The two explanatory figures use stylized synthetic examples and can be
generated directly:

```bash
python3 scripts/mainfigures_pipeline_concept.py
python3 scripts/mainfigures_dtw_topology.py
```

The scripts save PDFs in `figures/`. Forecast and evaluation figures use
pipeline outputs. To regenerate those, run:

```bash
make pipeline
make figures
```

The pipeline rerun can take several hours. Existing figures and analysis
outputs are included in the repository.

## Project layout

| Path | Contents |
|---|---|
| `tda_risk/` | Data processing, topology features, analog retrieval, forecasting, and evaluation code |
| `scripts/` | Analysis and figure-generation entry points |
| `data/` | Frozen aligned market-data panel and metadata |
| `results/` | Shipped analysis outputs |
| `figures/` | Generated PDF figures |
| `tests/` | Automated tests |

The final analysis uses `data/panel_published.csv.gz` and does not download or
refresh external data. Its date splits and forecast conventions are recorded
in [`revision_config/revision_protocol.json`](revision_config/revision_protocol.json).
The evaluation period is a fixed reanalysis panel, not a pristine untouched
holdout. See [`data/README.md`](data/README.md) for data sources and
[`docs/reproduction_notes.md`](docs/reproduction_notes.md) for additional
reproduction details.
To verify the shipped frozen analysis inputs, registries, and results, run:
