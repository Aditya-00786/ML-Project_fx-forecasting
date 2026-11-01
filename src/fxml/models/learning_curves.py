"""Generative against discriminative classifiers as the training window grows (RQ5).

Gaussian Naive Bayes estimates each feature's distribution within each class, a few numbers per
feature, so it should settle quickly. Logistic regression fits the decision boundary directly:
it should need more data but end up better. Both are trained on only the most recent ``n``
training days of each monthly fold and forecast the same test month. Every size therefore faces
the same test days, which keeps the comparison fair. Hyperparameters are fixed (no tuning), so
the curves show how each model learns rather than how well it was tuned.

Two targets are used. Next day direction carries almost no signal, so its curves mostly show
overfitting. A high volatility day (next day log realized variance above its median over the
last 63 days, a threshold known at the close) carries a strong signal, so its curves show
learning.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB

from fxml.models.inputs import BASE_FEATURES
from fxml.models.ml import imputed, scaled
from fxml.validation.metrics import expected_calibration_error
from fxml.validation.walkforward import Fold

SIZES = (21, 63, 126, 252, 504, 1008)  # one month to four years of trading days
FULL = 0  # size label for the whole expanding window
VOL_WINDOW = 63


def curve_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """The two binary targets: an up day, and a day more volatile than the recent median."""
    threshold = frame["logrv_d"].rolling(VOL_WINDOW).median()
    high = (frame["y_logrv"] > threshold).astype(float)
    return pd.DataFrame({
        "direction": frame["y_dir"],
        "high_volatility": high.where(frame["y_logrv"].notna() & threshold.notna()),
    })

CURVE_MODELS = {
    "naive_bayes": lambda: imputed(GaussianNB()),
    "logistic": lambda: scaled(LogisticRegression(max_iter=2000)),
}


def learning_curve_forecasts(
    frame: pd.DataFrame, folds: Sequence[Fold], sizes: Sequence[int] = SIZES
) -> pd.DataFrame:
    """Probability forecasts for every target, test day, model and training size.

    ``size`` 0 means the full expanding window; ``train_days`` is the number of days used.
    """
    targets = curve_targets(frame)
    x_all = frame[BASE_FEATURES].to_numpy(dtype=float)
    rows = []
    for fold in folds:
        for target in targets.columns:
            labels = targets[target].to_numpy()
            train = fold.train[~np.isnan(labels[fold.train])]
            test = fold.test[~np.isnan(labels[fold.test])]
            for size in [*sizes, FULL]:
                recent = train if size == FULL else train[-size:]
                if size > len(train) or len(test) == 0 or len(np.unique(labels[recent])) < 2:
                    continue
                for name, make in CURVE_MODELS.items():
                    model = make().fit(x_all[recent], labels[recent])
                    one = list(model.classes_).index(1)
                    rows.append(pd.DataFrame({
                        "target": target, "model": name, "size": size, "train_days": len(recent),
                        "date": frame.index[test], "actual": labels[test],
                        "forecast": model.predict_proba(x_all[test])[:, one],
                    }))
    return pd.concat(rows, ignore_index=True)


def summarise_curves(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, log loss, Brier score and AUC per target, model and size, pooled over pairs."""
    out = []
    for (target, model, size), group in forecasts.groupby(["target", "model", "size"]):
        y, p = group["actual"].to_numpy().astype(int), group["forecast"].clip(1e-6, 1 - 1e-6)
        out.append({
            "target": target, "model": model, "size": size,
            "train_days": group["train_days"].mean(),
            "days": len(group), "accuracy": np.mean((p > 0.5) == y),
            "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "roc_auc": roc_auc_score(y, p), "ece": expected_calibration_error(y, p),
        })
    return pd.DataFrame(out)
