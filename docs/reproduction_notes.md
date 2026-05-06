# Reproduction notes

Practical notes for running the pipeline end-to-end.

## Quick path

If you only need the figures and tables, the cached pipeline outputs
in `results/` are sufficient.

```bash
make figures
make tables
```

This finishes in a few minutes and never touches the network.

## Full pipeline

`make pipeline` reruns every backtest from scratch.  On a recent
laptop with `--n-jobs 4`, expect roughly:

| Stage | Time |
|---|---|
| Topology feature precomputation | ~25 s |
| Baseline pipeline (rolling HS, FHS, regime HS, GARCH, all $k$-NN baselines) | ~1.5 h |
| Topology pipeline | ~30 min |
| Hybrid scenario mixture | ~5 min |
| Formal backtests + table assembly | ~5 min |

Total wall clock: about 2–4 hours.

## Numerical determinism

- All forecasts are deterministic given the cached input panel.
- Random seeds are pinned in code: placebo permutation seed = 42,
  block-bootstrap seed = 12345.
- GARCH and FHS estimators are deterministic given the input data
  (no Monte Carlo step).

If you refresh the cache (`scripts/fetch_data.py --refresh`) after a
provider has revised historical data, downstream forecasts may
differ slightly from the manuscript.  The shipped cache reproduces
the manuscript exactly.

## Common issues

- **`yfinance` rate-limit on refresh**: if `--refresh` fails partway,
  rerun the script. yfinance occasionally throttles bulk historical
  downloads.
- **FRED `BAMLH0A0HYM2` short history**: this series only goes back
  to 2023 in the FRED snapshot. The pipeline transparently falls
  back to `BAA10Y` for earlier dates; both are dumped under
  `data/fred_raw/`.

## Output layout after `make all`

```
figures/fig{1..7}_*.pdf                      Regenerated figure PDFs
figures/table_mechanism_dtw_topology.csv
results/topology_pipeline/forecasts.csv      Refitted topology forecasts
results/baseline_pipeline/forecasts.csv      Refitted baseline forecasts
results/final_results_package/*.csv          Paper-facing tables
```
