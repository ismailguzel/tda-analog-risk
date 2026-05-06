# Makefile for the tda-analog-risk reproducibility package.
#
# Standard targets:
#
#   make figures   regenerate figure PDFs from cached results
#   make tables    regenerate paper-facing CSV tables
#   make pipeline  rerun the full backtest pipeline (~2-4 h)
#   make data      load or refresh cached data panel
#   make all       data + pipeline + figures + tables
#
# Override the Python interpreter via:
#
#   make figures PYTHON=/path/to/python

PYTHON ?= python

REPO_ROOT := $(shell pwd)

.PHONY: help data pipeline figures tables all

help:
	@echo "Targets:"
	@echo "  data       Load or refresh cached data panel"
	@echo "  pipeline   Run full backtest pipeline (heavy, ~2-4 h)"
	@echo "  figures    Regenerate figure PDFs from cached results"
	@echo "  tables     Regenerate paper-facing CSV tables"
	@echo "  all        data + pipeline + figures + tables"

data:
	$(PYTHON) scripts/fetch_data.py --use-cache

pipeline: data
	$(PYTHON) scripts/run_paper_pipeline.py --n-jobs 4

figures:
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/mainfigures_dtw_topology.py
	$(PYTHON) scripts/mainfigures_pipeline_concept.py

tables:
	$(PYTHON) scripts/build_final_results_package.py
	$(PYTHON) scripts/build_hybrid_comparison_table.py
	$(PYTHON) scripts/build_ablation_summary_table.py

all: data pipeline figures tables
