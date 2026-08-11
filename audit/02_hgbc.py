# %% [markdown]
# # Audit — 02_hgbc: HistGradientBoostingClassifier on baseline feature set
#
# Read-only review of the stored report below: its checks and metrics.

# %%
import skore
from skore import login

from titanic import PROJECT_ROOT

# %% [markdown]
# ## Open the project

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

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL printed by `put()`:
#   `https://skore.probabl.ai/skore-agent-yann/titanic/cross-validations/38084`
#
# Id: `skore:report:cross-validation:38084`

# %%
REPORT_ID = "skore:report:cross-validation:38084"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary
#
# Compare with 01_baseline: ROC-AUC 0.838 ± 0.015, Accuracy 0.788 ± 0.015.

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
