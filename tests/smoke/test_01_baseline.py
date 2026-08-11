"""Smoke test for `experiments/01_baseline.py`.

The Titanic baseline is an i.i.d. flat table with no cross-row feature
steps, so this is a sanity check rather than a cold-start diagnostic:
fit on one slice of the real data, predict on a disjoint slice, and
assert one prediction per predict-grid row. It still catches loader
row drops/duplication and NaN-poisoned predictions.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from titanic import PROJECT_ROOT
from titanic.data import TARGET_COL
from titanic.pipeline import build_learner

# Loose sanity bound (majority-class rate is ~0.62). Not a performance
# target — tighten against journal/01_baseline.md § Status.headline
# once the CV run is recorded.
MIN_SMOKE_ACCURACY = 0.6

N_PREDICT_ROWS = 120


@pytest.fixture
def train_predict_envs(tmp_path):
    """Return ``(train_env, predict_env, n_predict_rows, y_true)``.

    Splits the real Titanic CSV into a disjoint predict slice (the
    first ``N_PREDICT_ROWS`` rows) and a train slice (the rest), and
    writes each to its own temp directory so the pipeline's ``data_dir``
    binding is exercised exactly as in production.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest-provided per-test temporary directory.

    Returns
    -------
    train_env : dict
        Fit binding ``{"data_dir": <train dir>}``.
    predict_env : dict
        Predict binding ``{"data_dir": <predict dir>}``.
    n_predict_rows : int
        Number of rows in the predict slice.
    y_true : numpy.ndarray
        Ground-truth ``survived`` labels for the predict slice.
    """
    full = pd.read_csv(PROJECT_ROOT / "data" / "titanic.csv")
    predict_df = full.iloc[:N_PREDICT_ROWS].copy()
    train_df = full.iloc[N_PREDICT_ROWS:].copy()

    train_dir = tmp_path / "train"
    predict_dir = tmp_path / "predict"
    train_dir.mkdir()
    predict_dir.mkdir()
    train_df.to_csv(train_dir / "titanic.csv", index=False)
    predict_df.to_csv(predict_dir / "titanic.csv", index=False)

    train_env = {"data_dir": str(train_dir)}
    predict_env = {"data_dir": str(predict_dir)}
    y_true = predict_df[TARGET_COL].to_numpy()
    return train_env, predict_env, len(predict_df), y_true


def test_01_baseline(train_predict_envs):
    """One prediction per predict-grid row, and predictions are not degenerate."""
    train_env, predict_env, n_predict_rows, y_true = train_predict_envs

    learner = build_learner()
    learner.fit(train_env)
    predictions = learner.predict(predict_env)

    # HARD: structural correctness — one prediction per predict-grid row.
    assert len(predictions) == n_predict_rows, (
        f"got {len(predictions)} predictions for {n_predict_rows} "
        f"predict-grid rows — the loader is dropping or duplicating rows."
    )

    # SOFT: predictions are not NaN-poisoned or degenerate.
    smoke_accuracy = accuracy_score(y_true, predictions)
    assert smoke_accuracy > MIN_SMOKE_ACCURACY, (
        f"smoke accuracy {smoke_accuracy:.3f} <= {MIN_SMOKE_ACCURACY} — "
        f"predictions may be NaN-poisoned or degenerate."
    )
