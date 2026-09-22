"""Walk-forward backtesting engine."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.metaestimators import _safe_split

from skfin.estimators.mean_variance import MeanVariance


def compute_pnl(
    h: pd.DataFrame | pd.Series,
    ret: pd.DataFrame | pd.Series,
    pred_lag: int,
) -> pd.Series:
    """Compute PnL from holdings and returns.

    Args:
        h: Holdings at each time step.
        ret: Asset returns.
        pred_lag: Number of periods to lag holdings (avoids lookahead).

    Returns:
        PnL series (summed across assets if multi-asset).
    """
    pnl = h.shift(pred_lag).mul(ret)
    if isinstance(h, pd.DataFrame):
        pnl = pnl.sum(axis=1)
    return pnl


def fit_predict(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    return_estimator: bool = True,
) -> tuple:
    """Fit estimator on train split and predict on test split.

    Args:
        estimator: sklearn-compatible estimator.
        X: Feature matrix.
        y: Target matrix.
        train: Train indices.
        test: Test indices.
        return_estimator: If True, return (predictions, fitted_estimator).

    Returns:
        Predictions, or tuple of (predictions, estimator).
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)
    estimator.fit(X_train, y_train)
    if return_estimator:
        return estimator.predict(X_test), estimator
    else:
        return estimator.predict(X_test)


@dataclass
class Backtester:
    """Walk-forward backtester using sklearn TimeSeriesSplit."""

    estimator: BaseEstimator = None
    max_train_size: int = 36
    test_size: int = 1
    pred_lag: int = 1
    start_date: str = "1945-01-01"
    end_date: str | None = None
    name: str | None = None

    def __post_init__(self):
        if self.estimator is None:
            self.estimator = MeanVariance()

    def compute_holdings(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        pre_dispatch: str = "2*n_jobs",
        n_jobs: int = 1,
    ):
        """Compute holdings via walk-forward cross-validation.

        Args:
            X: Feature matrix with DatetimeIndex.
            y: Target matrix (returns used for covariance estimation).
            pre_dispatch: joblib pre_dispatch parameter.
            n_jobs: Number of parallel jobs.

        Returns:
            self (with h_, estimators_, cv_ fitted attributes).
        """
        X = X.loc[:self.end_date]
        cv = TimeSeriesSplit(
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            n_splits=1 + len(X.loc[self.start_date:]) // self.test_size,
        )
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        res = parallel(
            delayed(fit_predict)(
                clone(self.estimator), X.values, y.values, train, test, True
            )
            for train, test in cv.split(X)
        )
        y_pred, estimators = zip(*res)
        idx = X.index[np.concatenate([test for _, test in cv.split(X)])]
        if isinstance(y, pd.DataFrame):
            cols = y.columns
            h = pd.DataFrame(np.concatenate(y_pred), index=idx, columns=cols)
        elif isinstance(y, pd.Series):
            h = pd.Series(np.concatenate(y_pred), index=idx)
        else:
            h = None
        self.h_ = h
        self.estimators_ = estimators
        self.cv_ = cv
        return self

    def compute_pnl(self, ret: pd.DataFrame | pd.Series):
        """Compute PnL from fitted holdings and returns.

        Args:
            ret: Asset returns (same index/columns as y).

        Returns:
            self (with pnl_ fitted attribute).
        """
        pnl = compute_pnl(self.h_, ret, self.pred_lag)
        self.pnl_ = pnl.loc[self.start_date:self.end_date]
        if self.name:
            self.pnl_ = self.pnl_.rename(self.name)
        return self

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        ret: pd.DataFrame | pd.Series,
    ) -> pd.Series:
        """Run full backtest: compute holdings then PnL.

        Args:
            X: Feature matrix.
            y: Target matrix for fitting.
            ret: Returns for PnL computation.

        Returns:
            PnL series.
        """
        self.compute_holdings(X, y)
        self.compute_pnl(ret)
        return self.pnl_
