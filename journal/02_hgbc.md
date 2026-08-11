# 02_hgbc

## Question / hypothesis

Does replacing the logistic regression with a `HistGradientBoostingClassifier`
(the same model skore's SKD009 check uses as its strong baseline) improve
ROC-AUC significantly over the 01_baseline on the same feature set?

## Motivation

- **Sourcing strategy:** B1 (promoted from Backlog, sourced from `audit:01_baseline:checks.SKD009`)
- **Source(s):**
  - SKD009 issue in the 01_baseline report — "Test scores are not significantly
    better than a HistGradientBoosting baseline for 8/8 default predictive
    metrics." Docs: https://docs.skore.probabl.ai/0.23/user_guide/automated_checks.html#skd009-worse-than-baseline
  - Docs recommendation: "consider switching to a stronger default such as
    HistGradientBoostingClassifier".
- **Why this matters:** The 01_baseline (ROC-AUC 0.838 ± 0.015) sets the floor.
  If HGBC already beats it without any feature engineering, we learn the ceiling
  for the plain-feature regime and get a proper reference point for subsequent
  feature-engineering experiments.

## Method

- **Files touched:** `src/titanic/pipeline.py` (swap estimator only).
- **Change versus 01_baseline:** Replace `LogisticRegression(max_iter=1000)` with
  `HistGradientBoostingClassifier()` (skore's internal baseline; defaults only).
  Data loading, leakage-column exclusion, `skrub.tabular_pipeline` wrapping, and
  the `mark_as_X / mark_as_y` pattern are all unchanged. No new features.
- **Cross-validation:** same `KFold(n_splits=5, shuffle=True, random_state=0)`
  from `titanic.evaluate` — identical to 01_baseline for a direct comparison.
- **Out of scope:** feature engineering (`name`, `cabin`, `ticket`), hyperparameter
  tuning, calibration, ensemble stacking.

## Risks / things that could invalidate the result

- **Dataset size (1 309 rows):** HGBC is designed for larger datasets; defaults may
  over-regularise or under-regularise on this scale, making the comparison
  noisy rather than informative.
- **HGBC vs. skore's internal baseline are not identical runs:** skore refits its
  internal HGBC on each fold's train set independently; our CV does the same but
  is a separate fit with a fresh RNG state. Minor variance is expected even if
  the score is "the same".
- **`tabular_pipeline` double-wraps encoders:** skrub's tabular_pipeline already
  applies encoding/imputation; adding HGBC on top uses its native
  missing-value handling too. The interaction is benign but slightly redundant.
- **No tuning:** HGBC defaults (`max_iter=100`, `learning_rate=0.1`) may not be
  optimal for this dataset, so the result is a lower bound on what HGBC can do.

## Status

- **State:** done
- **Approved by user on:** 2026-08-11
- **Headline result:** ROC-AUC 0.858 ± 0.021 / Acc 0.804 ± 0.026 (5-fold KFold) — SKD009 still fires; new SKD001 overfitting (train/test gap on 9/9 metrics) + SKD010 ~24× slower fit
- **Implication for next iteration:** swapping the estimator gains ~2 pp ROC-AUC but introduces overfitting on this small dataset (1 309 rows) and doesn't clear SKD009. Feature engineering (title from `name`, deck from `cabin`) or regularisation tuning on HGBC are the natural next directions.
