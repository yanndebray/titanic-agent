# EDA — Titanic survival

_Generated from `data/eda.py` on 2026-07-10._

## Dataset at a glance

- **Tables:** one (`data/titanic.csv`, OpenML "titanic" v1)
- **Shape:** 1309 × 14
- **Target:** `survived` — binary classification (survived / did not)
- **Rich report:** [eda_titanic.html](eda_titanic.html)

## Per-column findings

| Column | Dtype | Missing | Note |
|---|---|---|---|
| `pclass` | Int64 | 0% | passenger class 1/2/3 — ordinal predictor |
| `survived` | Int64 | 0% | **target** |
| `name` | String | 0% | free text, unique per passenger (holds titles: Mr/Mrs/Miss) |
| `sex` | String | 0% | strong predictor (see Associations) |
| `age` | Float64 | **20.1%** | needs imputation |
| `sibsp` | Int64 | 0% | # siblings/spouses aboard |
| `parch` | Int64 | 0% | # parents/children aboard |
| `ticket` | String | 0% | high-cardinality id-like |
| `fare` | Float64 | 0.08% | one missing value |
| `cabin` | String | **77.5%** | mostly missing; deck letter recoverable but sparse |
| `embarked` | String | 0.15% | port C/Q/S |
| `boat` | String | 62.9% | **post-outcome — lifeboat number** |
| `body` | Float64 | **90.8%** | **post-outcome — body recovery id (only for the dead)** |
| `home.dest` | String | 43.1% | post-hoc research metadata; high missingness |

## Target

`survived` is binary with mean 0.382 — roughly **500 survivors / 809
deaths (~38% positive)**. Mildly imbalanced but not severe.

## Structure

No datetime columns and no natural grouping/entity key that repeats
across rows (`name`/`ticket` are near-unique per passenger, not group
keys spanning many rows). No temporal or grouped-CV structure — plain
random splitting is appropriate.

## Associations

`skrub.column_associations` (Cramér's V) surfaces:

- **`survived` ↔ `boat`: 0.948** — implausibly high. `boat` is the
  lifeboat number, known only *after* the outcome. **Leakage.**
- `survived` ↔ `sex`: 0.529 — the classic "women first" signal; the
  strongest *legitimate* predictor.
- `survived` ↔ `pclass`: 0.31 (pearson −0.31) — first class survived
  more.
- `fare` correlates with `pclass` (−0.56) — collinear proxies for
  wealth.

## Modelling implications

- **Drop the leakage columns `boat`, `body`, `home.dest`** before
  training. `boat`/`body` are recorded post-outcome; keeping them
  produces an artificially perfect model that cannot generalize to a
  real "will this passenger survive?" question.
- **Also drop `name` and `ticket`** for the baseline: near-unique
  free-text / id columns that a plain logistic regression can't use
  without overfitting (title extraction from `name` is a later
  experiment idea).
- **Mild class imbalance (~38% positive)** → use `StratifiedKFold` to
  keep fold class ratios stable; report **ROC-AUC** (skore's default
  for binary) rather than raw accuracy.
- **`age` missing 20%, one `fare` missing** → the pipeline needs
  imputation for numeric features.
- **Categoricals `sex`, `embarked`, `pclass`** → one-hot / skrub
  encoding; low cardinality, cheap.
- **`cabin` 77.5% missing** → drop for the baseline (deck extraction
  is a later idea).

## Open questions

- Confirm `boat`/`body`/`home.dest` are indeed post-outcome and should
  be excluded (assumed yes based on their definitions).
- Later experiments: extract title from `name`, deck letter from
  `cabin`, family-size from `sibsp`+`parch`.
