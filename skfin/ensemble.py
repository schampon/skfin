from dataclasses import dataclass

import numpy as np
import pandas as pd
from skfin.mv_estimators import Mbj
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class StackingBacktester:
    estimators: dict
    ret: pd.DataFrame
    max_train_size: int = 36
    test_size: int = 1
    start_date: str = "1945-01-01"
    end_date: str = None
    window: int = 60
    min_periods: int = 60
    final_estimator: BaseEstimator = Mbj()

    def __post_init__(self):
        self.ret = self.ret[: self.end_date]
        self.cv = TimeSeriesSplit(
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            n_splits=1
            + len(self.ret.loc[self.start_date : self.end_date]) // self.test_size,
        )

    def train(self, features, target):
        N_estimators = len(self.estimators)
        cols = self.ret.columns
        idx = self.ret.index[
            np.concatenate([test for _, test in self.cv.split(self.ret)])
        ]

        _h = {k: [] for k in list(self.estimators.keys()) + ["ensemble"]}
        _pnls = {k: [] for k in self.estimators.keys()}
        _coef = []
        for i, (train, test) in enumerate(self.cv.split(self.ret)):
            h_ = {}
            if i > self.min_periods:
                pnl_window = np.stack(
                    [np.array(v[-self.window :]) for k, v in _pnls.items()], axis=1
                )
                coef_ = self.final_estimator.fit(pnl_window).coef_
                _coef += [coef_]
            else:
                _coef += [np.zeros(N_estimators)]
            for k, m in self.estimators.items():
                m.fit(features.iloc[train], target.iloc[train])
                h_[k] = m.predict(features.iloc[test])
                _h[k] += [h_[k]]
                if i + 1 < len(idx):
                    _pnls[k] += [self.ret.loc[idx[i + 1]].dot(np.squeeze(h_[k]))]
            if i > self.min_periods:
                h_ensemble = (
                    np.stack([np.squeeze(v) for v in h_.values()], axis=1)
                    .dot(coef_)
                    .reshape(-1, 1)
                )
                V_ = m.named_steps["meanvariance"].V_
                h_ensemble = h_ensemble / np.sqrt(
                    np.diag(h_ensemble.T.dot(V_.dot(h_ensemble)))
                )
            else:
                h_ensemble = np.zeros([len(cols), 1])
            _h["ensemble"] += [h_ensemble.T]

        self.h_ = {
            k: pd.DataFrame(np.concatenate(_h[k]), index=idx, columns=cols)
            for k in _h.keys()
        }
        self.pnls_ = pd.concat(
            {
                k: v.shift(1).mul(self.ret).sum(axis=1)[self.start_date :]
                for k, v in self.h_.items()
            },
            axis=1,
        )
        self.coef_ = pd.DataFrame(
            np.stack(_coef), index=idx, columns=self.estimators.keys()
        )
        return self
