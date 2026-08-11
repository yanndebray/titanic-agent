# 01_baseline

## Question / hypothesis

Can a plain logistic regression on the basic passenger features
(class, sex, age, family size, fare, port) predict Titanic survival
with a useful ROC-AUC, giving us a first reference point?

## Motivation

- **Sourcing strategy:** my-pick:01_baseline (forced baseline — bootstrap)
- **Source(s):**
  - User request: "a short baseline with a logistic regression on the
    titanic".
  - EDA findings (`data/eda.md`): `sex` (Cramér's V 0.53) and `pclass`
    are the strongest legitimate predictors; three columns leak the
    outcome.
- **Why this matters:** every later experiment (title extraction,
  gradient boosting, feature engineering) is judged against this
  reference. It also pins down the leakage-column exclusion up front.

## Method

- **Files touched:** `src/titanic/data.py`, `src/titanic/pipeline.py`,
  `src/titanic/evaluate.py`, `experiments/01_baseline.py`,
  `tests/smoke/test_01_baseline.py`.
- **Change versus baseline:** this *is* the baseline.
- **Data:** load `data/titanic.csv`. Target = `survived`.
  **Drop the leakage columns** `boat`, `body`, `home.dest`, and the
  near-unique id/text columns `name`, `ticket`, `cabin` (deferring
  title/deck extraction to a later experiment).
- **Pipeline:** a skrub DataOps graph — skrub's default table
  preprocessing (encoding of `sex`/`embarked`/`pclass`, numeric
  imputation of `age`/`fare`, scaling) feeding
  `sklearn.linear_model.LogisticRegression`. Mechanics owned by
  `build-ml-pipeline`.
- **Cross-validation:** `KFold(n_splits=5, shuffle=True,
  random_state=0)` (chosen at G-CV-SPLITTER). No group/time structure
  → plain K-fold; stratification is deliberately avoided even for the
  mild imbalance (it produces over-confident error bars).
- **Metric:** skore's binary-classification defaults (ROC-AUC as the
  headline); no override.
- **Out of scope:** feature engineering (titles, deck, family size),
  hyperparameter tuning, non-linear models.

## Risks / things that could invalidate the result

- **Leakage** — if `boat`/`body`/`home.dest` are not dropped, ROC-AUC
  will be near-perfect and meaningless. The smoke test + explicit drop
  guard against this.
- **`age` 20% missing** — imputation strategy affects the score; the
  baseline uses skrub's default, not a tuned choice.
- **Small dataset (1309 rows)** — fold-to-fold variance may be
  non-trivial; read the ROC-AUC with its spread, not as a point value.
- **Dropping `name`/`ticket`/`cabin`** discards real signal (titles,
  deck); the baseline deliberately underuses the data.

## Status

- **State:** approved
- **Approved by user on:** 2026-07-10
- **Headline result:** n/a
- **Implication for next iteration:** n/a
