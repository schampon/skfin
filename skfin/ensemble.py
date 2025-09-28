from dataclasses import dataclass

import numpy as np
import pandas as pd
from skfin.mv_estimators import Mbj
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class StackingBacktester:
    estimators: dict
    max_train_size: int = 36
    test_size: int = 1
    start_date: str = "1945-01-01"
    end_date: str = None
    window: int = 60
    min_periods: int = 60
    final_estimator: BaseEstimator = Mbj()

    def train(self, X, y, ret):
        cv = TimeSeriesSplit(
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            n_splits=1 + len(X.loc[self.start_date : self.end_date]) // self.test_size,
        )
        N_estimators = len(self.estimators)
        cols = X.columns
        idx = X.index[np.concatenate([test for _, test in cv.split(X)])]

        _h = {k: [] for k in list(self.estimators.keys()) + ["ensemble"]}
        _next_pnls = {k: [] for k in self.estimators.keys()}
        _coef = []
        for i, (train, test) in enumerate(cv.split(X)):
            h_ = {}
            # each strategy position and next-period pnls 
            for k, m in self.estimators.items():
                m.fit(X.iloc[train], y.iloc[train])
                h_[k] = m.predict(X.iloc[test])
                _h[k] += [h_[k]]
                if i + 1 < len(idx):
                    _next_pnls[k] += [ret.loc[idx[i + 1]].dot(np.squeeze(h_[k]))]
            # compute coef from strategy pnls   
            if i <= self.min_periods:
                _coef += [np.zeros(N_estimators)]
            else:
                pnl_window = np.stack(
                    [np.array(v[-self.window-1 :-1]) for k, v in _next_pnls.items()], axis=1
                )
                coef_ = self.final_estimator.fit(pnl_window).coef_
                _coef += [coef_]                      
            # ensemble 
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
            _h["ensemble"] += [h_ensemble.T]
            
        self.h_ = {
            k: pd.DataFrame(np.concatenate(_h[k]), index=idx, columns=cols)
            for k in _h.keys()
        }
        self.pnls_ = pd.concat(
            {
                k: v.shift(1).mul(ret).sum(axis=1)[self.start_date :]
                for k, v in self.h_.items()
            },
            axis=1,
        )
        self.coef_ = pd.DataFrame(
            np.stack(_coef), index=idx, columns=self.estimators.keys()
        )
        self.cv = cv 
        return self
