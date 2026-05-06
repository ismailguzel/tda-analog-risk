# Formal Backtest Notes

Inputs:
- forecasts file: `results/topology_pipeline/forecasts.csv`
- split: `test`

Outputs:
- `formal_backtests_by_method.csv`
- `fz_style_ranking.csv`
- `dm_vs_reference.csv`
- `mcs_style_survivors.csv`
- `mcs_style_trace.csv`

Important:
- Kupiec and Christoffersen are run on VaR(99%) exceedances.
- The joint score is an FZ-style surrogate for project convention
  VaR(99%) + ES(97.5%) (mixed alpha levels).
- `mcs_style_survivors.csv` is a bootstrap elimination set, intended as an
  MCS-style ranking aid for this stage.
