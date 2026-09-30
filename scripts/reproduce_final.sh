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

BIBTEX="${BIBTEX_BIN:-bibtex}"

hash_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

verify_existing() {
  "$PYTHON" scripts/verify_final_reproduction.py
  "$PYTHON" scripts/verify_response_page_lines.py
  pdfinfo manuscript/revised_source/main.pdf | grep -E '^Pages:'
  pdfinfo manuscript/revised_source/main_latexdiff.pdf | grep -E '^Pages:'
  pdfinfo response/final/response_to_editor_and_reviewers.pdf | grep -E '^Pages:'
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
7. final table, figure, manuscript, latexdiff, and response-letter builds
EOF
}

compile_article() {
  local source_dir="$1"
  local tex_name="$2"
  local output_pdf="$3"
  local build_dir
  build_dir="$(mktemp -d "${TMPDIR:-/tmp}/tda-article.XXXXXX")"
  cp -R "$source_dir"/. "$build_dir"/
  (
    cd "$build_dir"
    pdflatex -interaction=nonstopmode -halt-on-error "$tex_name" >/dev/null
    "$BIBTEX" "${tex_name%.tex}" >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "$tex_name" >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "$tex_name" >/dev/null
  )
  cp "$build_dir/${tex_name%.tex}.pdf" "$output_pdf"
}

compile_latexdiff() {
  local build_dir
  build_dir="$(mktemp -d "${TMPDIR:-/tmp}/tda-latexdiff.XXXXXX")"
  cp -R manuscript/revised_source/. "$build_dir"/
  (
    cd "$build_dir"
    pdflatex -interaction=nonstopmode -halt-on-error main_latexdiff.tex >/dev/null
    "$BIBTEX" main_latexdiff >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error main_latexdiff.tex >/dev/null
    # Deleted passages in the diff still refer to labels that existed only in
    # the submitted manuscript.  Seed those original label values so a clean
    # rebuild does not display unresolved references in deleted text.
    cat >> main_latexdiff.aux <<'EOF'
\newlabel{tab:datastats}{{1}{11}{Descriptive statistics for daily portfolio losses in the test period (2015--2026), reported in log-return units}{table.1}{}}
\newlabel{fig:landscape}{{3}{12}{Weighted H1 persistence landscapes for three representative dates}{figure.3}{}}
\newlabel{eq:landscape}{{12}{12}{Topology pipeline for the return path}{equation.12}{}}
\newlabel{tab:baseline_grid}{{2}{15}{Test-period performance of baseline families}{table.2}{}}
\newlabel{tab:head_to_head}{{3}{16}{Test-period performance of validation-selected specifications}{table.3}{}}
\newlabel{tab:formal_tests}{{4}{16}{Formal backtest results}{table.4}{}}
\newlabel{tab:hybrid}{{5}{17}{Matched-date comparison of FHS-EWMA, topology, and the scenario mixture}{table.5}{}}
\newlabel{fig:var_ts}{{4}{18}{Realized daily portfolio loss and VaR forecasts}{figure.4}{}}
\newlabel{fig:monthly_exc}{{5}{19}{Monthly VaR exceedance rates}{figure.5}{}}
\newlabel{tab:stress}{{6}{20}{Stress-period results}{table.6}{}}
\newlabel{fig:covid_case}{{6}{21}{Daily realized losses and VaR forecasts during the COVID crash}{figure.6}{}}
\newlabel{sec:baseline_grid_appendix}{{A.1}{23}{Baseline grids and pooled family counts}{subsection.1.A.1}{}}
\newlabel{tab:ablation_alpha}{{A1}{23}{State-blending coefficient}{table.1}{}}
\newlabel{tab:ablation_steps}{{A2}{24}{Discretization grid sensitivity}{table.2}{}}
\newlabel{fig:w_sensitivity}{{A1}{27}{Scenario-mixture weight sensitivity}{figure.1}{}}
EOF
    pdflatex -interaction=nonstopmode -halt-on-error main_latexdiff.tex >/dev/null
  )
  cp "$build_dir/main_latexdiff.pdf" manuscript/revised_source/main_latexdiff.pdf
}

compile_response() {
  local build_dir
  build_dir="$(mktemp -d "${TMPDIR:-/tmp}/tda-response.XXXXXX")"
  cp response/final/response_to_editor_and_reviewers.tex "$build_dir/"
  (
    cd "$build_dir"
    pdflatex -interaction=nonstopmode -halt-on-error response_to_editor_and_reviewers.tex >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error response_to_editor_and_reviewers.tex >/dev/null
  )
  cp "$build_dir/response_to_editor_and_reviewers.pdf" response/final/response_to_editor_and_reviewers.pdf
}

build_only() {
  local old_main old_response
  old_main="$(hash_file manuscript/revised_source/main.pdf 2>/dev/null || true)"
  old_response="$(hash_file response/final/response_to_editor_and_reviewers.pdf 2>/dev/null || true)"

  "$PYTHON" scripts/build_corrected_manuscript_materials.py
  "$PYTHON" scripts/mainfigures_pipeline_concept.py --output-dir manuscript/revised_source/figures
  "$PYTHON" scripts/mainfigures_dtw_topology.py --output-dir manuscript/revised_source/figures
  latexdiff --exclude-textcmd='section,subsection,subsubsection,caption' \
    --disable-citation-markup \
    manuscript/submitted_source/main.tex manuscript/revised_source/main.tex \
    > manuscript/revised_source/main_latexdiff.tex
  perl -pi -e 's/[ \t]+$//' manuscript/revised_source/main_latexdiff.tex
  compile_article manuscript/revised_source main.tex manuscript/revised_source/main.pdf
  compile_latexdiff
  compile_response

  local new_main new_response
  new_main="$(hash_file manuscript/revised_source/main.pdf)"
  new_response="$(hash_file response/final/response_to_editor_and_reviewers.pdf)"
  printf '%s\n' "[OK] Manuscript PDF: $new_main (previous: ${old_main:-none}; rebuild differences are intentional)"
  printf '%s\n' "[OK] Response PDF: $new_response (previous: ${old_response:-none}; rebuild differences are intentional)"
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
    verify_existing
    ;;
  --plan)
    print_plan
    ;;
  *)
    printf '%s\n' 'Usage: scripts/reproduce_final.sh --verify-existing|--build-only|--full|--plan' >&2
    exit 2
    ;;
esac
