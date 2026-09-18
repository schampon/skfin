"""Sklearn estimator wrappers with transform() for pipeline compatibility."""

from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression as _LinearRegression,
    Ridge as _Ridge,
    RidgeCV as _RidgeCV,
)
from sklearn.multioutput import MultiOutputRegressor as _MultiOutputRegressor
from sklearn.neural_network import MLPRegressor as _MLPRegressor


def _add_transform(cls):
    """Decorator that aliases transform to predict."""
    def transform(self, X):
        return self.predict(X)
    cls.transform = transform
    return cls


@_add_transform
class LinearRegression(_LinearRegression):
    """LinearRegression with transform() for sklearn pipeline use."""
    pass


@_add_transform
class Ridge(_Ridge):
    """Ridge with transform() for sklearn pipeline use."""
    pass


@_add_transform
class RidgeCV(_RidgeCV):
    """RidgeCV with transform() for sklearn pipeline use."""
    pass


@_add_transform
class MLPRegressor(_MLPRegressor):
    """MLPRegressor with transform() for sklearn pipeline use."""
    pass


@_add_transform
class MultiOutputRegressor(_MultiOutputRegressor):
    """MultiOutputRegressor with transform() for sklearn pipeline use."""
    pass


class MultiLGBMRegressor(BaseEstimator):
    """Multi-output LightGBM regressor with transform() for pipeline use."""

    def __init__(self, **kwargs):
        kwargs["n_jobs"] = 1
        kwargs.setdefault("verbose", -1)
        kwargs.setdefault("min_data_in_bin", 1)
        self._kwargs = kwargs
        self.m = None

    def get_params(self, deep: bool = True) -> dict:
        return self._kwargs.copy()

    def set_params(self, **kwargs):
        kwargs["n_jobs"] = 1
        self._kwargs = kwargs
        return self

    def fit(self, X, y):
        from lightgbm.sklearn import LGBMRegressor

        self.m = _MultiOutputRegressor(LGBMRegressor(**self._kwargs))
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict(X)

    def transform(self, X):
        return self.predict(X)
