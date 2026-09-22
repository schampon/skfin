"""Britten-Jones (1999) mean-variance weight estimator."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


@dataclass
class Mbj(BaseEstimator):
    """Compute unconstrained mean-variance weights via the Britten-Jones (1999) trick.

    Regresses a vector of ones on asset returns to recover optimal weights.
    Equivalent to solving for the tangency portfolio without explicit covariance inversion.

    Args:
        positive: If True, constrain weights to be non-negative.
        rescaled_positions: If True, normalize weights to unit norm.
    """

    positive: bool = False
    rescaled_positions: bool = True

    def fit(self, X, y=None):
        m = LinearRegression(fit_intercept=False, positive=self.positive)
        m.fit(X, y=np.ones(len(X)))
        if self.rescaled_positions:
            self.coef_ = m.coef_ / np.sqrt(np.sum(m.coef_**2))
        else:
            self.coef_ = m.coef_
        return self

    def transform(self, X):
        """Project returns onto optimal weights to produce portfolio PnL."""
        return X.dot(self.coef_)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def predict(self, X):
        return np.array([self.coef_])
