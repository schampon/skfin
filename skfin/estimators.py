from lightgbm.sklearn import LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor


class LinearRegression(LinearRegression):
    def transform(self, X):
        return self.predict(X)


class Ridge(Ridge):
    def transform(self, X):
        return self.predict(X)


class RidgeCV(RidgeCV):
    def transform(self, X):
        return self.predict(X)


class MLPRegressor(MLPRegressor):
    def transform(self, X):
        return self.predict(X)


class MultiOutputRegressor(MultiOutputRegressor):
    def transform(self, X):
        return self.predict(X)


class MultiLGBMRegressor(BaseEstimator):
    """
    Multi-output extension of the lightgbm regressor as a transform class
    get_params and set_params attributes necessary for cloning the class
    """

    def __init__(self, **kwargs):
        if "n_jobs" in kwargs.keys():
            kwargs["n_jobs"] = 1
        else:
            kwargs = {"n_jobs": 1, **kwargs}
        self.m = MultiOutputRegressor(LGBMRegressor(**kwargs))

    def get_params(self, deep=True):
        return self.m.estimator.get_params(deep=deep)

    def set_params(self, **kwargs):
        if "n_jobs" in kwargs.keys():
            kwargs["n_jobs"] = 1
        else:
            kwargs = {"n_jobs": 1, **kwargs}
        return self.m.estimator.set_params(**kwargs)

    def fit(self, X, y):
        return self.m.fit(X, y)

    def transform(self, X):
        return self.m.transform(X)
