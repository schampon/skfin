from lightgbm.sklearn import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor


def add_transform_method(cls):
    """
    Decorator to add a 'transform' method to a class that uses the 'predict' method.
    """
    def transform(self, X):
        return self.predict(X)
        
    cls.transform = transform
    return cls

@add_transform_method
class LinearRegression(LinearRegression):
    pass

@add_transform_method
class Ridge(Ridge):
    pass

@add_transform_method
class RidgeCV(RidgeCV):
    pass

@add_transform_method
class MLPRegressor(MLPRegressor):
    pass

@add_transform_method
class MultiOutputRegressor(MultiOutputRegressor):
    pass


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
