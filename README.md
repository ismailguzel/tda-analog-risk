# Clean final revision package

This archive contains the frozen panel, valid registries, final results, the
canonical Python package, tests, the revised manuscript, the latexdiff PDF,
and the final point-by-point response letter.

The default reproduction path never downloads or refreshes external financial
data. It uses `data/panel_published.csv.gz`; the panel hash, row count, and date
range are recorded in `revision_config/revision_protocol.json`.

## Commands

From the extracted package root:

```bash
python -m pip install -r requirements.txt
./scripts/reproduce_final.sh --verify-existing
./scripts/reproduce_final.sh --build-only
./scripts/reproduce_final.sh --plan
./scripts/reproduce_final.sh --full
```

`--verify-existing` is the fast integrity check, including the topology
window-length table, cutoff-tie diagnostic, and every response-letter
page/line citation. `--build-only` regenerates
final table fragments, figures, the revised manuscript PDF, the latexdiff PDF,
and the response-letter PDF. `--plan` prints the complete expensive-analysis
execution graph without running it. `--full` runs the complete expensive analysis,
including the locked null design at radii 0, 125, and 250, the validation
k-sensitivity design at radii 0 and 125, and the two explicitly post hoc
mechanism diagnostics: a 500-draw age-stratified recency benchmark and a
100-window temporal-order shuffle analysis.
The exact installed environment used for the frozen outputs is recorded in
`environment/revision_pip_freeze.txt`.

The cutoff diagnostic checks the distance at ranks $k$ and $k+1$ for the
locked evaluation and validation sensitivity catalogs. Exact ties are handled
in the direct-permutation pilot by a seeded random key independent of candidate
date and feature origin; the computational uniform-subset implementation targets the
same tie-invariant exchangeability-based distribution.


## Post hoc mechanism diagnostics

Two exploratory diagnostics are shipped separately from the prespecified model
selection and fixed-panel comparison.  The recency-matched benchmark repeats the
no-exclusion retrieval comparison 500 times while matching, at every forecast
date, the topology neighbor counts in seven prespecified age bands.  The
temporal-order diagnostic preserves each sampled 250-day window's empirical
marginal distribution but shuffles its order before recomputing the weighted
$H_1$ landscape.  It uses 100 evenly spaced evaluation windows, 25 shuffles per
window, and a deterministic 30-point subsample of each delay-coordinate cloud.
These analyses diagnose mechanism; they are not additional model-selection or
confirmatory forecast comparisons.

## System prerequisites

Python 3.9.6 is the documented environment. Install Python packages from
`requirements.txt`. Document rebuilding additionally requires TeX Live (or an
equivalent LaTeX distribution) with `pdflatex` and `bibtex` (or `bibtex8` by setting `BIBTEX_BIN=bibtex8`), `latexdiff` with
Perl, and Poppler's `pdfinfo` and `pdftotext` for verification. The verification
scripts also require `shasum` or an equivalent SHA-256 utility. These are system tools, not
Python packages. The default path uses the frozen panel and never downloads
external financial data.

The package excludes Git metadata, virtual environments, caches, auxiliary
LaTeX products, archived/superseded result trees, malformed registries, and
live-download utilities. See `provenance/result_hash_manifest.sha256` for the
hashes of the frozen panel and shipped final outputs.
