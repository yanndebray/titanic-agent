"""Data loading for the Titanic survival experiments.

Reads the single OpenML "titanic" table and returns it with the
post-outcome leakage columns and the near-unique id/text columns
removed. The EDA (`data/eda.md`) established that `boat`, `body`, and
`home.dest` are recorded *after* the outcome and would leak the
target, and that `name`/`ticket`/`cabin` are near-unique or mostly
missing and are dropped for the baseline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from titanic import PROJECT_ROOT

TARGET_COL = "survived"

# Recorded after the outcome — keeping any of these leaks the target.
LEAKAGE_COLS = ["boat", "body", "home.dest"]

# Near-unique ids / free text / mostly-missing — dropped for the
# baseline (title/deck extraction is deferred to a later experiment).
DROP_COLS = ["name", "ticket", "cabin"]


def load_raw(data_dir: str | Path) -> pd.DataFrame:
    """Load the Titanic table from ``data_dir`` with leaky columns removed.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``titanic.csv``.

    Returns
    -------
    pandas.DataFrame
        The passenger table including the ``survived`` target, without
        the leakage or dropped columns.
    """
    df = pd.read_csv(Path(data_dir) / "titanic.csv")
    return df.drop(columns=LEAKAGE_COLS + DROP_COLS)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for the Titanic survival task.

    Returns
    -------
    X : pandas.DataFrame
        Feature columns (target removed).
    y : pandas.Series
        The binary ``survived`` target.
    """
    df = load_raw(PROJECT_ROOT / "data")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
