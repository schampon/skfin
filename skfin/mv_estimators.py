import numpy as np
import pandas as pd
from skfin.metrics import sharpe_ratio
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

from dataclasses import dataclass, field
from typing import Callable, Optional


def compute_batch_holdings(pred, V, A=None, risk_target=None):
    """
    compute markowitz holdings with return prediction "mu" and covariance matrix "V"

    Args: 
        pred: (numpy.ndarray or pandas.Series or pandas.DataFrame): Expected returns, can be of shape (N,) or (N, K).
        V: (numpy.ndarray): Covariance matrix of shape (N, N).
        A: (numpy.ndarray, optional): Matrix for linear constraints, default is None
        past_h (numpy.ndarray, optional): Not used in the current implementation.
        constant_risk (bool, optional): If True, normalize outputs to maintain constant risk, default is False.

    Returns:
        numpy.ndarray: Computed holdings.
    """

    N, _ = V.shape
    if isinstance(pred, (pd.Series, pd.DataFrame)):
        pred = pred.values
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    elif pred.shape[1] == N:
        pred = pred.T

    invV = np.linalg.inv(V)
    if A is None:
        M = invV
    else:
        U = invV.dot(A)
        if A.ndim == 1:
            M = invV - np.outer(U, U.T) / U.dot(A)
        else:
            M = invV - U.dot(np.linalg.inv(U.T.dot(A)).dot(U.T))
    h = M.dot(pred)
    if risk_target is not None:
        h = risk_target * h / np.sqrt(np.diag(h.T.dot(V.dot(h))))
    return h.T



@dataclass
class MeanVariance(BaseEstimator):
    """
    A mean-variance optimization estimator that computes portfolio holdings 
    based on expected returns and the covariance matrix.

    Attributes:
        transform_V (Callable): Function to transform target variable 'y' into a covariance matrix.
        A (Optional[np.ndarray]): Constraints matrix for the optimization problem.
        risk_target (float): Risk target for the portfolio.
    """
    transform_V: Callable = field(default=lambda x: np.cov(x.T))
    A: Optional[np.ndarray] = None
    risk_target: float = 1.0

    def __post_init__(self):
        """
        Post-initialization process to set additional attributes or setup.
        """
        self.holdings_kwargs = {'risk_target': self.risk_target}
    
    @staticmethod
    def compute_batch_holdings(pred, V, A, risk_target, **kwargs):
        """
        Compute portfolio holdings in a batch manner.

        Parameters:
            pred (np.ndarray): Predicted returns.
            V (np.ndarray): Covariance matrix.
            A (np.ndarray): Constraint matrix.
            risk_target (float): Target risk level.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Portfolio holdings.
        """
        return compute_batch_holdings(pred=pred, V=V, A=A, **kwargs)
    
    def fit(self, X, y=None):
        """
        Fit the model by calculating the covariance matrix 'V_' from targets 'y'.

        Parameters:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target variable matrix.
        """
        self.V_ = self.transform_V(y)

    def predict(self, X, **kwargs):
        """
        Predict portfolio holdings based on input features.

        Parameters:
            X (np.ndarray): Input feature matrix.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Predicted portfolio holdings.
        """
        A = self.A if self.A is not None else np.ones(X.shape[1])
        kwargs = {**kwargs, **self.holdings_kwargs}
        h = self.compute_batch_holdings(pred=X, V=self.V_, A=A, **kwargs)
        return h
        
    def score(self, X, y):
        """
        Calculate the performance score of the portfolio using Sharpe ratio.

        Parameters:
            X (np.ndarray): Predicted returns.
            y (np.ndarray): Actual returns.

        Returns:
            float: Sharpe ratio of the portfolio.
        """
        return sharpe_ratio(np.sum(X * y, axis=1))


class Mbj(TransformerMixin):
    """
    Computing unconstrained mean-variance weights with the Britten-Jones (1999) trick.
    """

    def __init__(self, positive=False):
        self.positive = positive

    def fit(self, X, y=None):
        m = LinearRegression(fit_intercept=False, positive=self.positive)
        m.fit(X, y=np.ones(len(X)))
        self.coef_ = m.coef_ / np.sqrt(np.sum(m.coef_**2))
        return self

    def transform(self, X):
        return X.dot(self.coef_)


class TimingMeanVariance(BaseEstimator):
    def __init__(self, transform_V=None, a_min=None, a_max=None):
        if transform_V is None:
            self.transform_V = lambda x: np.var(x)
        else:
            self.transform_V = transform_V
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        self.V_ = self.transform_V(y)

    def predict(self, X):
        if (self.a_min is None) & (self.a_max is None):
            h = X / self.V_
        else:
            h = np.clip(
                X / np.sqrt(self.V_), a_min=self.a_min, a_max=self.a_max
            ) / np.sqrt(self.V_)
        return h
