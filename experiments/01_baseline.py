# %% [markdown]
# # Experiment 01: logistic-regression baseline
#
# **Date:** 2026-07-10
# **Goal:** establish a first survival-prediction reference with a plain
# logistic regression on the basic passenger features (leakage columns
# dropped). See `journal/01_baseline.md`.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from titanic import PROJECT_ROOT
from titanic.data import load_dataset
from titanic.evaluate import splitter
from titanic.pipeline import build_learner

# %% [markdown]
# ## Paths
#
# `PROJECT_ROOT` resolves from the installed package, independent of the
# current working directory. `DATA_DIR` holds `titanic.csv`.

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project
#
# Reports persist to the Skore Hub workspace `skore-agent-yann` under a
# stable key (the file stem). `login(mode="hub")` is interactive on the
# first run and cached afterwards.

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
# `data_dir_preview=DATA_DIR` makes `learner.skb.preview()` work; it does
# not affect what `skore.evaluate` fits on (that comes from `data=`).

# %%
X, y = load_dataset()
learner = build_learner(data_dir_preview=DATA_DIR)

# %% [markdown]
# ## Evaluate
#
# The `SkrubLearner` takes a single environment dict, so the source
# binding is passed via `data=`. skore selects the binary-classification
# metrics (ROC-AUC headline) automatically; the cross-validator comes
# from `titanic.evaluate`.

# %%
report = skore.evaluate(
    learner,
    data={"data_dir": str(DATA_DIR)},
    splitter=splitter,
)
report

# %% [markdown]
# ## Persist
#
# Key = file stem. Reusing this key overwrites the stored report — fork
# into a new experiment file to keep both.

# %%
project.put("01_baseline", report)
