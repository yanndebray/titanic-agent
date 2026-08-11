"""Inputs to `skore.evaluate` for the Titanic baseline.

Holds the cross-validator passed to `skore.evaluate(...)`. The EDA
found no group or temporal structure, so plain (shuffled) K-fold is
the right choice. Stratification is deliberately avoided despite the
mild class imbalance — it compresses across-fold variance and yields
over-confident error bars. skore picks the binary-classification
metrics (ROC-AUC, accuracy, precision, recall, F1) automatically.
"""

from __future__ import annotations

from sklearn.model_selection import KFold

splitter = KFold(n_splits=5, shuffle=True, random_state=0)
