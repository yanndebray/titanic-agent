# %% [markdown]
# # Audit — 01_baseline: logistic regression on basic passenger features
#
# Read-only review of the stored report below: its checks and metrics.

# %%
import skore
from skore import login

from titanic import PROJECT_ROOT

# %% [markdown]
# ## Open the project
#
# Open the same project the experiment wrote to. The init block below
# matches `experiments/01_baseline.py` exactly.

# %%
login(mode="hub")
project = skore.Project(
    name="titanic",
    mode="hub",
    workspace="skore-agent-yann",
)
project

# %% [markdown]
# ## List the available reports
#
# `project.summarize()` provides an overview of all reports in this
# Project — useful to confirm the experiment's report landed and to
# spot duplicate keys from accidental re-runs.

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL printed by `put()`:
#   `https://skore.probabl.ai/skore-agent-yann/titanic/cross-validations/38070`
#
# Type segment is `cross-validations` (plural) → id uses `cross-validation`
# (singular). Id: `skore:report:cross-validation:38070`.

# %%
REPORT_ID = "skore:report:cross-validation:38070"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary
#
# Each row carries a `code`, a `severity` (`passed` / `issue` / `tip`),
# and a `documentation_url` — the linked page describes what the check
# tests and what to try next.

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary
#
# Binary classification defaults: accuracy, precision, recall, F1,
# ROC-AUC, log-loss + fit/predict timings. Mean ± std across the 5 folds.

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
#
# This file is the durable record of how the experiment's report was
# reviewed; re-run it any time to refresh the checks and metrics above.
