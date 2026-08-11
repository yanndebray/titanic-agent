# %% [markdown]
# # Experiment 02: HistGradientBoostingClassifier
#
# **Date:** 2026-08-11
# **Goal:** swap the logistic regression for a HistGradientBoostingClassifier
# (skore's SKD009 internal baseline) on the same feature set and see if it
# clears the SKD009 check. See `journal/02_hgbc.md`.
# **Result:** filled in after the run.

# %%
import skore
from sklearn.ensemble import HistGradientBoostingClassifier
from skore import login

from titanic import PROJECT_ROOT
from titanic.data import load_dataset
from titanic.evaluate import splitter
from titanic.pipeline import build_learner

# %% [markdown]
# ## Paths

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project

# %%
login(mode="hub")
project = skore.Project(
    name="titanic",
    mode="hub",
    workspace="skore-agent-yann",
)

# %% [markdown]
# ## Data and learner
#
# Same feature set and preprocessing as `01_baseline`. Only the tail
# estimator changes: `HistGradientBoostingClassifier()` with all defaults.

# %%
X, y = load_dataset()
learner = build_learner(
    data_dir_preview=DATA_DIR,
    estimator=HistGradientBoostingClassifier(),
)

# %% [markdown]
# ## Evaluate

# %%
report = skore.evaluate(
    learner,
    data={"data_dir": str(DATA_DIR)},
    splitter=splitter,
)
report

# %% [markdown]
# ## Persist

# %%
project.put("02_hgbc", report)
