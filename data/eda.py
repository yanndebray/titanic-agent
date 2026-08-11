# %% [markdown]
# # EDA: Titanic survival
#
# Exploratory data analysis of the Titanic dataset, run before designing
# a model.
#
# - **Raw data** is read-only — `RAW` points at `data/titanic.csv`. This
#   file never cleans or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (`titanic/data/`): an
#   `eda_titanic.html` report, summarized in `eda.md`.

# %%
import json

import skrub

from titanic import PROJECT_ROOT

# EDA outputs always land here (created if missing); the raw data may
# live elsewhere.
EDA_DIR = PROJECT_ROOT / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# The single Titanic table (OpenML v1), 1309 passengers.

# %%
import pandas as pd

RAW = pd.read_csv(PROJECT_ROOT / "data" / "titanic.csv")
RAW.shape

# %% [markdown]
# ## Table overview
#
# Per-table report (column types, distributions, associations) saved to
# `data/eda_titanic.html`, plus a compact per-column summary: dtype,
# fraction missing, and number of unique values.

# %%
report = skrub.TableReport(RAW, title="titanic", verbose=0)
report.write_html(EDA_DIR / "eda_titanic.html")

summary = json.loads(report.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("nunique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}

# %% [markdown]
# ## Target
#
# The target's distribution — class balance for `survived`. This shapes
# the metric and whether cross-validation should stratify.

# %%
TARGET = "survived"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)

# %% [markdown]
# ## Structure signals
#
# Datetime columns (which point to time-based validation) and
# high-cardinality id / group-like columns (which point to grouped
# validation, to avoid leaking an entity across folds).

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("nunique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}

# %% [markdown]
# ## Associations
#
# Strongest pairwise column associations. Strong feature-target links
# are candidate predictors; an implausibly perfect one is a possible
# leakage flag to call out explicitly.

# %%
skrub.column_associations(RAW).head(20)

# %% [markdown]
# ## Summary
#
# The findings and their modelling implications are written up in
# `data/eda.md`.
