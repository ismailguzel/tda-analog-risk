from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical paper pipeline: baseline, topology finalist, "
            "hybrids, formal backtests, and paper-facing tables."
        )
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use for subprocess runs.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker count for topology feature precomputation.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline pipeline run.",
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="Skip topology finalist pipeline run.",
    )
    parser.add_argument(
        "--skip-hybrids",
        action="store_true",
        help="Skip both hybrid runs.",
    )
    parser.add_argument(
        "--skip-formal-tests",
        action="store_true",
        help="Skip formal backtest generation.",
    )
    parser.add_argument(
        "--skip-packaging",
        action="store_true",
        help="Skip final package and comparison-table generation.",
    )
    parser.add_argument(
        "--skip-placebo-test",
        action="store_true",
        help="Skip placebo significance rebuild.",
    )
    parser.add_argument(
        "--skip-ablation-summary",
        action="store_true",
        help="Skip ablation summary rebuild.",
    )
    parser.add_argument(
        "--skip-appendix",
        action="store_true",
        help="Skip appendix experiment reruns (k-sweep, alpha/steps, feature, multiseries).",
    )
    parser.add_argument(
        "--skip-validation-finalists",
        action="store_true",
        help="Skip validation-based finalist selection rebuild.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove canonical result directories/files before rerunning the paper pipeline.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"\n[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _remove_path(path: Path) -> None:
    target = REPO_ROOT / path
    if not target.exists():
        return
    print(f"[clean] removing {path}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def main() -> None:
    args = parse_args()
    py = args.python

    topology_output = Path("results/topology_pipeline")
    baseline_output = Path("results/baseline_pipeline")
    scenario_cov_output = Path("results/topology_fhs_scenario_mixture")
    scenario_wc_output = Path("results/topology_fhs_scenario_mixture_weightedcombo")
    hybrid_output = Path("results/topology_fhs_hybrid")
    appendix_outputs = [
        Path("results/topology_k_sweep"),
        Path("results/topology_landscape_alpha_steps_ablation"),
        Path("results/topology_landscape_feature_ablation"),
        Path("results/topology_multiseries_ablation"),
    ]
    validation_finalists_output = Path("results/validation_finalists")
    package_output = Path("results/final_results_package")
    hybrid_table_output = Path("results/hybrid_comparison_table.csv")

    if args.clean:
        for path in [
            baseline_output,
            topology_output,
            hybrid_output,
            scenario_cov_output,
            scenario_wc_output,
            validation_finalists_output,
            package_output,
            hybrid_table_output,
            *appendix_outputs,
        ]:
            _remove_path(path)

    if not args.skip_baseline:
        _run([py, "scripts/run_baseline_pipeline.py"])

    if not args.skip_topology:
        _run(
            [
                py,
                "scripts/run_topology_pipeline.py",
                "--n-jobs",
                str(max(args.n_jobs, 1)),
                "--output-dir",
                str(topology_output),
            ]
        )

    if not args.skip_hybrids:
        _run(
            [
                py,
                "scripts/run_topology_fhs_hybrid.py",
                "--forecasts",
                str(topology_output / "forecasts.csv"),
                "--output-dir",
                str(hybrid_output),
                "--selection-objective",
                "coverage_then_fz",
                "--target-var-exceedance",
                "0.01",
            ]
        )
        _run(
            [
                py,
                "scripts/run_topology_fhs_scenario_mixture.py",
                "--output-dir",
                str(scenario_cov_output),
                "--selection-objective",
                "coverage_then_fz",
                "--target-var-exceedance",
                "0.01",
                "--n-jobs",
                str(max(args.n_jobs, 1)),
            ]
        )
        _run(
            [
                py,
                "scripts/run_topology_fhs_scenario_mixture.py",
                "--output-dir",
                str(scenario_wc_output),
                "--selection-objective",
                "weighted_combo",
                "--target-var-exceedance",
                "0.01",
                "--coverage-weight",
                "0.7",
                "--n-jobs",
                str(max(args.n_jobs, 1)),
            ]
        )

    if not args.skip_formal_tests:
        _run(
            [
                py,
                "scripts/run_formal_backtests.py",
                "--forecasts",
                str(topology_output / "forecasts.csv"),
                "--output-dir",
                str(topology_output / "formal_tests"),
            ]
        )
        _run(
            [
                py,
                "scripts/run_formal_backtests.py",
                "--forecasts",
                str(scenario_cov_output / "hybrid_forecasts.csv"),
                "--output-dir",
                str(scenario_cov_output / "formal_tests"),
            ]
        )
        _run(
            [
                py,
                "scripts/run_formal_backtests.py",
                "--forecasts",
                str(scenario_wc_output / "hybrid_forecasts.csv"),
                "--output-dir",
                str(scenario_wc_output / "formal_tests"),
            ]
        )

    if not args.skip_appendix:
        _run([py, "scripts/run_topology_k_sweep.py"])
        _run([py, "scripts/run_topology_landscape_alpha_steps_ablation.py"])
        _run([py, "scripts/run_topology_landscape_feature_ablation.py"])
        _run([py, "scripts/run_topology_multiseries_ablation.py"])

    if not args.skip_validation_finalists:
        _run(
            [
                py,
                "scripts/select_validation_finalists.py",
                "--forecasts",
                str(topology_output / "forecasts.csv"),
                "--output-dir",
                str(validation_finalists_output),
            ]
        )

    if not args.skip_placebo_test:
        _run(
            [
                py,
                "scripts/run_placebo_significance_test.py",
                "--forecasts",
                str(topology_output / "forecasts.csv"),
                "--output",
                "results/final_results_package/table_placebo_significance.csv",
            ]
        )

    if not args.skip_ablation_summary:
        _run([py, "scripts/build_ablation_summary_table.py"])

    if not args.skip_packaging:
        _run(
            [
                py,
                "scripts/build_hybrid_comparison_table.py",
                "--topology-forecasts",
                str(topology_output / "forecasts.csv"),
                "--output",
                "results/final_results_package/table_hybrid_comparison_matched.csv",
            ]
        )
        _run([py, "scripts/build_final_results_package.py"])

    print("\nPaper pipeline completed.")


if __name__ == "__main__":
    main()
