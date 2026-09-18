"""Rolling ensemble backtester for combining multiple strategies."""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit

from skfin.estimators.mbj import Mbj

logger = logging.getLogger(__name__)


@dataclass
class StackingBacktester:
    """Walk-forward ensemble that learns strategy combination weights over rolling windows.

    Fits multiple estimators independently, then learns combination weights
    from their out-of-sample PnLs using a final estimator (default: Britten-Jones).

    Args:
        estimators: Dict mapping strategy names to sklearn-compatible estimators.
        max_train_size: Maximum training window for each strategy.
        test_size: Number of periods per test fold.
        start_date: Start date for PnL output.
        end_date: End date for PnL output.
        window: Rolling window size for learning combination weights.
        min_periods: Minimum periods before learning weights (outputs zeros before).
        final_estimator: Estimator that learns combination weights from PnLs.
    """

    estimators: dict = field(default_factory=dict)
    max_train_size: int = 36
    test_size: int = 1
    start_date: str = "1945-01-01"
    end_date: str | None = None
    window: int = 60
    min_periods: int = 60
    final_estimator: BaseEstimator = field(default_factory=Mbj)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        ret: pd.DataFrame,
    ):
        """Run walk-forward ensemble backtest.

        Args:
            X: Feature matrix with DatetimeIndex.
            y: Target matrix for fitting individual strategies.
            ret: Returns for PnL computation.

        Returns:
            self (with h_, pnls_, coef_ fitted attributes).
        """
        cv = TimeSeriesSplit(
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            n_splits=1 + len(X.loc[self.start_date:self.end_date]) // self.test_size,
        )
        n_estimators = len(self.estimators)
        cols = X.columns
        idx = X.index[np.concatenate([test for _, test in cv.split(X)])]

        _h = {k: [] for k in list(self.estimators.keys()) + ["ensemble"]}
        _next_pnls = {k: [] for k in self.estimators.keys()}
        _coef = []

        for i, (train, test) in enumerate(cv.split(X)):
            h_ = {}
            for k, m in self.estimators.items():
                m.fit(X.iloc[train], y.iloc[train])
                h_[k] = m.predict(X.iloc[test])
                _h[k].append(h_[k])
                if i + 1 < len(idx):
                    _next_pnls[k].append(ret.loc[idx[i + 1]].dot(np.squeeze(h_[k])))

            if i <= self.min_periods:
                _coef.append(np.zeros(n_estimators))
            else:
                pnl_window = np.stack(
                    [np.array(v[-self.window - 1:-1]) for v in _next_pnls.values()],
                    axis=1,
                )
                coef_ = self.final_estimator.fit(pnl_window).coef_
                _coef.append(coef_)

            if i <= self.min_periods:
                h_ensemble = np.zeros([len(cols), 1])
            else:
                h_ensemble = (
                    np.stack([np.squeeze(v) for v in h_.values()], axis=1)
                    .dot(coef_)
                    .reshape(-1, 1)
                )
                V_ = m.named_steps["meanvariance"].V_
                h_ensemble = h_ensemble / np.sqrt(
                    np.diag(h_ensemble.T.dot(V_.dot(h_ensemble)))
                )
            _h["ensemble"].append(h_ensemble.T)

        self.h_ = {
            k: pd.DataFrame(np.concatenate(_h[k]), index=idx, columns=cols)
            for k in _h
        }
        self.pnls_ = pd.concat(
            {
                k: v.shift(1).mul(ret).sum(axis=1)[self.start_date:]
                for k, v in self.h_.items()
            },
            axis=1,
        )
        self.coef_ = pd.DataFrame(
            np.stack(_coef), index=idx, columns=list(self.estimators.keys())
        )
        self.cv_ = cv
        return self
