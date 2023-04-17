import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.metaestimators import _safe_split


class Backtester:
    def __init__(
        self,
        estimator,
        ret,
        max_train_size=36,
        test_size=1,
        start_date="1945-01-01",
        end_date=None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.estimator = estimator
        self.ret = ret[: self.end_date]
        self.cv = TimeSeriesSplit(
            max_train_size=max_train_size,
            test_size=test_size,
            n_splits=1 + len(ret.loc[start_date:end_date]) // test_size,
        )

    def train(self, features, target):
        pred, estimators = fit_predict(
            self.estimator, features, target, self.ret, self.cv, return_estimator=True
        )
        self.estimators_ = estimators
        self.h_ = pred
        if isinstance(pred, pd.DataFrame):
            self.pnl_ = (
                pred.shift(1).mul(self.ret).sum(axis=1)[self.start_date : self.end_date]
            )
        elif isinstance(pred, pd.Series):
            self.pnl_ = pred.shift(1).mul(self.ret)[self.start_date : self.end_date]
        return self


def _fit_predict(estimator, X, y, train, test, return_estimator=False):
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)
    estimator.fit(X_train, y_train)
    if return_estimator:
        return estimator.predict(X_test), estimator
    else:
        return estimator.predict(X_test)


def fit_predict(
    estimator,
    features,
    target,
    ret,
    cv,
    return_estimator=False,
    verbose=0,
    pre_dispatch="2*n_jobs",
    n_jobs=1,
):
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    res = parallel(
        delayed(_fit_predict)(
            clone(estimator), features, target, train, test, return_estimator
        )
        for train, test in cv.split(ret)
    )
    if return_estimator:
        pred, estimators = zip(*res)
    else:
        pred = res

    idx = ret.index[np.concatenate([test for _, test in cv.split(ret)])]
    if isinstance(ret, pd.DataFrame):
        cols = ret.columns
        df = pd.DataFrame(np.concatenate(pred), index=idx, columns=cols)
    elif isinstance(ret, pd.Series):
        df = pd.Series(np.concatenate(pred), index=idx)
    else:
        df = None

    if return_estimator:
        return df, estimators
    else:
        return df
