# Incremental value and fixed-prediction diagnostics

This extension asks whether frozen representations add predictive information to
the gravity design, not whether a representation-only readout beats gravity.
It reuses the existing reciprocal-dyad splits and representation caches. No LLM
inference is performed. Original standalone-model implementations are unchanged.

## Models and fitting

| Identifier | Unpenalized block | Penalized block |
|---|---|---|
| G_BASE | Intercept; log1p training-only origin/hour and destination/hour totals; log1p distance | None |
| G_NAME | Same gravity design | Frozen district-name representation |
| G_COORD | Same gravity design | Frozen coordinate representation |
| G_POLY2 | Same gravity design | Coordinate–arrival-hour polynomial features |

Training totals are corrected for retained district-pair coverage. Training rows
contribute to their own training totals; held-out targets do not. The response is
log1p mobility. The objective is SSE + alpha times the squared coefficient norm
of the added block; gravity coefficients are unpenalized. Float64 SVD solves the
added block after projection against the gravity design. Normal-equation residuals
and a synthetic augmented-least-squares comparison check the solver.

Within each outer training set, the saved inner split selects from 13 penalties
(10^-6 through 10^6) and a base-only candidate using validation log-MSE. Ties use
rtol=1e-10 and atol=1e-12, favoring base-only and then the larger penalty.
Preprocessing and coefficients are refitted on outer training data after selection.
Each of the 14,400 observations receives one held-out prediction per model.
Prediction on the flow scale is max(0, expm1(predicted log flow)).

Added features are standardized using the relevant training subset only. The
polynomial condition starts with four historical prompt coordinates (rounded to
three decimals), sin(2*pi*h/24), and cos(2*pi*h/24), where h is arrival hour.
The six standardized features expand to 27 degree-two features, standardized again
on training data. There is no additional scaling after projection against gravity.

## Inputs and commands

Run from the repository root with `PYTHONPATH=src`. This extension was verified
using the historical Python 3.11.11 environment in `requirements-incremental.txt`.
Use a separate environment and execute the source scripts (not `pip install .`,
whose existing package metadata requires Python >=3.12):

```sh
python3.11 -m venv .venv-incremental
source .venv-incremental/bin/activate
python -m pip install -r requirements-incremental.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

The extension has not been retested against the original release's newer pinned
environment. The original requirements and package metadata remain unchanged.
The fitting script uses the Unix `resource` module for runtime reporting.
These are fixed-study replay scripts, not arbitrary-dataset estimators. Required
inputs are not distributed here; raw downloads alone do not recreate the caches.

Fitting requires:

- An inputs directory with `mobility.parquet`, `prompt_source.parquet`, `folds.csv`,
  and `inner_splits.csv` (see [input provenance](INPUTS.md)).
- A reference-results directory with `oof_predictions_all_models.parquet` and
  `bin_assignments.csv`, including the original gravity predictions for verification.
- A cache directory with `original/` and `geometry/`, containing the preserved
  representation arrays and row indices checked against `core.EXPECTED_CACHE_SHA256`.

```sh
PYTHONPATH=src python scripts/fit_incremental.py \
  --inputs-dir /path/to/accepted_inputs \
  --reference-results-dir /path/to/reference_results \
  --cache-dir /path/to/analysis_cache --output-dir /path/to/NEW_fit
```

`--preflight` checks inputs, caches, the synthetic solver, and gravity reproduction
without fitting the three augmented conditions. Use a separate fresh output folder
for the later full run. No command overwrites an existing output directory.

To evaluate already saved predictions without fitting:

```sh
PYTHONPATH=src python scripts/evaluate_incremental.py \
  --input-dir /path/to/completed_fit --output-dir /path/to/NEW_evaluation
PYTHONPATH=src python scripts/diagnose_incremental.py \
  --input-dir /path/to/NEW_evaluation --output-dir /path/to/NEW_diagnostics
python scripts/verify_incremental_diagnostics.py --output-dir /path/to/NEW_diagnostics
```

The evaluation input needs `data/observations.csv`, `data/inner_splits.csv`,
`results/oof_predictions.csv`, and `results/tuning_results.csv`. Observations have
`obs_id, origin_code, destination_code, hour, unordered_pair_id, y_reference,
distance_km, outer_fold, flow_decile, distance_quintile`. Predictions have
`obs_id, model_id, outer_fold, predicted_log_flow, predicted_flow, selected_alpha,
base_only`. Tuning contains all 14 candidates per augmented model and fold.
Study-size, key, fold, selected-penalty, finite-value, and transformation checks
are executable assertions: do not run with `python -O`.

## Outputs and interpretation

Evaluation produces pooled/fold metrics, six pairwise comparisons, selected
settings, district total/allocation diagnostics, fixed flow/distance/hour groups,
independent metric checks, and two 300-dpi review figures with source tables.
`paired_comparisons.csv` uses candidate minus comparator; smaller is better for
error metrics, larger for CPC. Read signed bias relative to zero.

Follow-up diagnostics produce observation/dyad errors, gain/loss concentration,
influential-pair summaries, common-pair exclusion sensitivity, group/fold
comparisons, contributions, flow-by-distance overlap, and two review figures
(PNG/PDF). For error metrics, positive `improvement` means comparator error minus
candidate error; CPC improvement is candidate minus comparator. Bias columns in
`group_fold_comparisons.csv` have explicit meanings:

- `signed_bias_difference_pp`: comparator signed bias minus candidate signed
  bias. Its sign describes a shift, not whether the candidate is closer to zero.
- `absolute_bias_improvement_pp`: absolute comparator bias minus absolute
  candidate bias. Positive means closer to zero, negative means farther from
  zero, and zero means equal absolute bias, including opposite signs.
- `improvement_total_bias_pct`: deprecated compatibility alias of
  `signed_bias_difference_pp`. Historical values are preserved; do not use its
  sign as an improvement decision.

Both new columns are percentage-point differences of percentage biases, not
relative percentage reductions. Undefined bias remains NaN. These columns do
not change the MAE/RMSE/sMAPE fold-consistency counts or the evaluation script's
absolute-bias criterion. MAE and MSE contributions are additive; MSE contributions are not RMSE
contributions. Loss concentration uses the explicitly reported loss denominator,
not total error. The exclusion diagnostic is post hoc and does not replace full
sample performance or justify deleting observations. Counts of improving folds
are descriptive, not significance tests.

`verify_incremental_diagnostics.py` independently recomputes diagnostic quantities.
Optional `--replicate-dir /path/to/prior_diagnostics` additionally compares saved
CSV outputs (rtol=1e-10, atol=1e-7). For legacy outputs without the two explicit
bias columns, every historical column is compared; the new columns are checked
independently against fixed predictions. Its report does not claim visual figure review.
These scripts generate diagnostic figures, not the final manuscript's edited
figure layout or numbering. Dataset-specific prose reports are not generated.

Only code and documentation are public. Generated outputs include research data
and local provenance paths; review their redistribution eligibility separately.
No raw data, prompt records, embeddings, OOF predictions, or private reports are
included in this repository. See [data access](DATA_DOWNLOAD.md).

## Release verification (2026-10-15)

The 12 data-free tests passed, including the added solver and training-only
polynomial-scaling checks. Input/cache checks, synthetic solver validation, and
gravity reproduction passed through the public preflight entry point. Full
augmented fitting was not repeated during this code release. Evaluation and
follow-up diagnostics were rerun from saved OOF predictions and compared with
the original outputs; 1,301 independent diagnostic checks passed. This does not
establish raw-data-to-LLM reproduction or independent empirical replication.
