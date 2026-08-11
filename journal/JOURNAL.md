# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog — keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** `titanic` — binary classification (survival)
- **Goal:** predict passenger survival on the Titanic dataset; establish a logistic-regression baseline.
- **Last experiment:** 02_hgbc — done
- **Last result:** ROC-AUC 0.858 ± 0.021 | Accuracy 0.804 ± 0.026 | ⚠️ SKD001 overfitting + SKD009 still fires + SKD010 slow fit

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas — recorded: 2026-07-10
  - env manager: uv — recorded: 2026-07-10
  - agent feature: installed — recorded: 2026-07-10
  - optional features: none — recorded: 2026-07-10
  - package name (`src/<pkg>/`): titanic — recorded: 2026-07-10
  - skore mode: hub — recorded: 2026-07-10
  - skore hub workspace: skore-agent-yann — recorded: 2026-07-10
  - skore mlflow tracking uri: n/a — recorded: 2026-07-10
  - CV splitter family: <decided at the evaluate step of 01_baseline>

## Data understanding (EDA)

- **Status:** done — 2026-07-10
- **Summary:** 1309 × 14, target `survived` ~38% positive (mild
  imbalance). `age` missing 20%, `fare` 1 value. Critically, `boat`
  (Cramér's V 0.948 with survived), `body`, and `home.dest` are
  post-outcome leakage columns and must be dropped; `name`/`ticket`
  are near-unique ids dropped for the baseline. `sex` and `pclass`
  are the strongest legitimate predictors. No datetime/group
  structure → plain stratified CV.
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| 01_baseline | logistic regression on basic features (leakage cols dropped) | done | ROC-AUC 0.838 ± 0.015 / Acc 0.788 — SKD009 ⚠️ | [design note](01_baseline.md) |
| 02_hgbc | swap estimator to HistGradientBoostingClassifier, same feature set | done | ROC-AUC 0.858 ± 0.021 / Acc 0.804 — SKD001 🚨 SKD009 ⚠️ SKD010 ⚠️ | [design note](02_hgbc.md) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Swap logistic regression for `HistGradientBoostingClassifier` wrapped in `skrub.tabular_pipeline` — the same model skore's SKD009 check uses as its strong baseline; the current LR fails to beat it on all 8 metrics | `audit:01_baseline:checks.SKD009` |
| B2 | Engineer title from `name` and deck from `cabin` before feeding into the logistic regression — the EDA flagged these columns as carrying real predictive signal that the baseline deliberately discarded | `audit:01_baseline:checks.SKD009` |
