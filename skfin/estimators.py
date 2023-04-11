from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV


class LinearRegression(LinearRegression):
    def transform(self, X):
        return self.predict(X)


class Ridge(Ridge):
    def transform(self, X):
        return self.predict(X)


class RidgeCV(RidgeCV):
    def transform(self, X):
        return self.predict(X)


class MultiOutputRegressor(MultiOutputRegressor):
    def transform(self, X):
        return self.predict(X)


class MLPRegressor(MLPRegressor):
    def transform(self, X):
        return self.predict(X)
