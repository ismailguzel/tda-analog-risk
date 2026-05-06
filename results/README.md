# Results

Precomputed pipeline outputs.

## Subdirectories

| Path | Purpose |
|---|---|
| `topology_pipeline/` | The retained $w$-Topology finalist. `forecasts.csv` holds the daily VaR/ES forecasts and per-method exceedance flags; `summary.csv` aggregates by method/split. |
| `baseline_pipeline/` | All classical and analog baselines (Rolling HS, FHS-EWMA, Regime-HS, GARCH-$t$, $s$-Euclidean, $s$-Mahalanobis, $w$-Euclidean, $w$-DTW, $w$-FPCA, Random $k$-NN). |
| `topology_fhs_scenario_mixture/` | Scenario-level mixture of the topology pipeline with FHS-EWMA. Contains the matched-date hybrid forecasts and validation weight scan. |
| `final_results_package/` | Paper-facing tables (CSVs) and the formal-test outputs. |
| `ablation_summaries/` | Summary CSVs from appendix ablations (state-blending coefficient, discretization grid, k-sweep, feature mode, multi-series input). The full per-row forecasts are too large to ship; only the summaries are included. |

## Reproducing

```bash
make pipeline   # rerun baseline + topology + hybrid pipelines (~2-4 h)
make tables     # rebuild paper-facing CSV tables
```

`make tables` operates on the already-computed `forecasts.csv` files,
so it is fast (< 1 min). `make pipeline` is the heavy step and refits
every model from scratch using the cached panel under `data/`.

## Key tables

| File | Description |
|---|---|
| `final_results_package/table_test_baseline_grid.csv` | Pooled baseline grid summary |
| `final_results_package/table_test_finalist_head_to_head.csv` | Validation-selected head-to-head |
| `final_results_package/formal_tests/formal_backtests_by_method.csv` | Kupiec, Christoffersen, FZ-style scores |
| `final_results_package/table_main_hybrid_comparison.csv` | Matched-date hybrid (FHS / Topology / Mixture) |
| `final_results_package/table_stress_period_summary.csv` | Stress-period disaggregation |
| `topology_pipeline/forecasts.csv` | All $w$-Topology forecasts (test panel) |
| `topology_fhs_scenario_mixture/hybrid_forecasts.csv` | Scenario-mixture forecasts |
