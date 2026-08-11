"""Learner declaration for the Titanic survival baseline.

Declares a skrub DataOps graph: load the passenger table, mark the
feature/target split, and fit a logistic regression on top of skrub's
default tabular preprocessing (which encodes the low-cardinality
categoricals ``sex``/``embarked``/``pclass`` and imputes the missing
``age``/``fare`` values). This is an i.i.d. flat table — no cross-row
feature steps — so the X marker sits directly on the loaded frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import skrub
from sklearn.linear_model import LogisticRegression

from titanic.data import TARGET_COL, load_raw


def build_learner(
    data_dir_preview: str | Path | None = None,
    estimator: Any = None,
):
    """Return the unfit learner (a skrub ``SkrubLearner``).

    Parameters
    ----------
    data_dir_preview : str or Path or None, optional
        Preview binding for the ``data_dir`` source var, used only by
        ``learner.skb.preview()`` during interactive iteration. Leave
        as ``None`` for fit / cross-validate — the env-dict passed to
        ``skore.evaluate(..., data={"data_dir": ...})`` supplies the
        binding.
    estimator : sklearn-compatible estimator or None, optional
        Final estimator passed to ``skrub.tabular_pipeline``. Defaults
        to ``LogisticRegression(max_iter=1000)`` — preserves the
        ``01_baseline`` behaviour for callers that do not pass this
        argument.
    """
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000)

    data_dir = (
        skrub.var("data_dir", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("data_dir")
    )

    data = data_dir.skb.apply_func(load_raw)
    X = data.drop(columns=[TARGET_COL]).skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    predictions = X.skb.apply(
        skrub.tabular_pipeline(estimator),
        y=y,
    )
    return predictions.skb.make_learner()
