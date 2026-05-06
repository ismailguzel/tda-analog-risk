# Final Results Package Notes

This folder contains paper-facing artifacts generated from:

- `results/baseline_pipeline`
- `results/topology_pipeline`

Key outputs:

- `table_main_selected_models.csv`: main-text test table using validation-selected benchmark specifications.
- `table_main_hybrid_comparison.csv`: main-text matched-date comparison for topology, FHS, and scenario mixture.
- `figure_main_monthly_selected_models.csv`: reduced monthly panel for the main text figure.
- `figure_main_covid_selected_models.csv`: reduced COVID-window panel for the main text figure.
- `table_test_baseline_grid.csv`: aggregate test metrics from the full baseline grid.
- `table_test_finalist_head_to_head.csv`: pooled test metrics from the retained topology run (appendix/supporting only).
- `table_test_focus_methods.csv`: focused table for topology/placebo/core baselines.
- `table_stress_period_summary.csv`: method metrics inside predefined stress windows.
- `figure_monthly_exceedance_panel.csv`: monthly exceedance panel for plotting.
- `case_study_covid_window_daily.csv`: day-level panel for qualitative stress-case reading.
- `formal_tests/`: canonical topology formal-test exports copied from `results/topology_pipeline/formal_tests`.
- `package_metadata.json`: run window and source-artifact summary for the package build.

Notes:

- `docs/arxiv/mainfigures.py` generates a synthetic mechanism figure for exposition; it is intentionally kept separate from the empirical package.
- `window_euclidean_knn` is a supporting raw-window $L^2$ comparator added to separate window-level retrieval effects from topology-specific representation effects; it is not included in the current main-text selected-model table by default.
