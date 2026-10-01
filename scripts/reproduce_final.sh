#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Keep figure generation self-contained in clean extraction tests and on
# systems where the user's default matplotlib cache is not writable.
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
  MPLCONFIGDIR="$(mktemp -d "${TMPDIR:-/tmp}/tda-mpl-cache.XXXXXX")"
  export MPLCONFIGDIR
fi
if [[ -z "${MPLBACKEND:-}" ]]; then
  export MPLBACKEND=Agg
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

verify_existing() {
  "$PYTHON" scripts/verify_final_reproduction.py "$@"
}

print_plan() {
  cat <<'EOF'
Canonical full reproduction plan (dry run; no scientific computation executed)
1. blocked validation over the prespecified candidate universe
2. locked evaluation for the 13 finalist families at radii 0, 125, and 250
3. repeated history-only nulls:
   - topology history-feature permutation (conceptual placebo)
   - analytically equivalent uniform-subset retrieval (computational cross-check)
   A. locked evaluation null analysis: radii 0, 125, and 250; 500 draws per
      implementation and radius
   B. validation k-sensitivity: radii 0 and 125; k in {100, 250, 500, 750,
      1000}; 250 draws per implementation, radius, and k
4. paired proper-loss inference and standard radius-specific MCS
5. exclusion, representation, cutoff-tie, recency-matched, and temporal-order diagnostics
6. topology window-length sensitivity table from frozen validation outputs
7. figure builds (fig1 pipeline concept, fig2 DTW-versus-topology mechanism)
EOF
}

# Rebuild the derived outputs that need no new scientific computation: the
# window-length sensitivity table and the two explanatory figures.
build_only() {
  "$PYTHON" scripts/build_window_sensitivity_table.py
  "$PYTHON" scripts/mainfigures_pipeline_concept.py --output-dir figures
  "$PYTHON" scripts/mainfigures_dtw_topology.py --output-dir figures
}

full_pipeline() {
  "$PYTHON" scripts/run_revision_validation.py
  "$PYTHON" scripts/run_locked_evaluation.py
  "$PYTHON" scripts/run_final_nulls.py
  "$PYTHON" scripts/run_final_inference.py
  "$PYTHON" scripts/run_final_diagnostics.py
  "$PYTHON" scripts/check_cutoff_ties.py
  "$PYTHON" scripts/run_posthoc_diagnostics.py
  build_only
}

case "${1:-}" in
  --verify-existing)
    verify_existing
    ;;
  --build-only)
    verify_existing
    build_only
    verify_existing
    ;;
  --full)
    full_pipeline
    # Recomputed outputs carry fresh timing fields, so check schemas and
    # designs rather than byte-level hashes.
    verify_existing --skip-hash-check
    ;;
  --plan)
    print_plan
    ;;
  *)
    printf '%s\n' 'Usage: scripts/reproduce_final.sh --verify-existing|--build-only|--full|--plan' >&2
    exit 2
    ;;
esac
