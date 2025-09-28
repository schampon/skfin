from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import TimeSeriesSplit
from skfin.mv_estimators import MeanVariance 


def compute_pnl_components(h, ret, vol_liquidity_factor=None):
    ret = ret[h.index[0] : h.index[-1]]
    vol_liquidity_factor = vol_liquidity_factor.loc[h.index[0] : h.index[-1]]

    pnl = h.shift(1).mul(ret).sum(axis=1)
    if vol_liquidity_factor is not None:
        impact_cost = h.diff().pow(2).mul(vol_liquidity_factor).sum(axis=1)
        return {
            "gross": pnl,
            "net = gross - impact cost": pnl.sub(impact_cost),
            "impact cost": -1 * impact_cost,
        }
    else:
        return pnl


def compute_batch_holdings_with_cost(
    pred, V, A=None, past_h=None, vol_liquidity_factor=None, lambda_=None, risk_target=None
):
    """
    compute markowitz holdings with return prediction "mu" and covariance matrix "V"

    mu: numpy array (shape N * K)
    V: numpy array (N * N)

    """
    if (lambda_ is None) & (vol_liquidity_factor is not None):
        lambda_ = 1
    N, _ = V.shape
    if isinstance(pred, pd.Series) | isinstance(pred, pd.DataFrame):
        pred = pred.values
    if pred.shape == (N,):
        pred = pred[:, None]
    elif pred.shape[1] == N:
        pred = pred.T

    if vol_liquidity_factor is not None:
        invV = np.linalg.inv(V / lambda_ + 2 * np.diag(vol_liquidity_factor))
    else:
        invV = np.linalg.inv(V)
    if A is None:
        M = invV
    else:
        U = invV.dot(A)
        if A.ndim == 1:
            M = invV - np.outer(U, U.T) / U.dot(A)
        else:
            M = invV - U.dot(np.linalg.inv(U.T.dot(A)).dot(U.T))
    if (vol_liquidity_factor is not None) & (past_h is not None):
        h = M.dot(pred + 2 * np.diag(vol_liquidity_factor).dot(past_h.T))
    else:
        h = M.dot(pred)
    return h.T


@dataclass
class MeanVarianceWithCost(MeanVariance):
    """
    Mean-variance optimization estimator with transaction cost considerations.
    """
    @staticmethod
    def compute_batch_holdings(pred, V, A, **kwargs):
        """
        Compute portfolio holdings considering transaction costs.

        Parameters:
            pred (np.ndarray): Predicted returns.
            V (np.ndarray): Covariance matrix.
            A (np.ndarray): Constraint matrix.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Portfolio holdings considering costs.
        """
        return compute_batch_holdings_with_cost(pred, V, A, **kwargs)


@dataclass 
class BacktesterWithCost:
    estimator: BaseEstimator = MeanVarianceWithCost()
    vol_liquidity_factor: pd.DataFrame=None
    max_train_size: int=36
    test_size: int=1
    start_date: str="1945-01-01"
    end_date: str=None
    h_init: pd.Series=None
    return_pnl_component: bool=False

    def train(self, X, y, ret):
        X = X.loc[:self.end_date]
        cv = TimeSeriesSplit(
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            n_splits=1 + len(X.loc[self.start_date : ]) // self.test_size,
        )
        
        _h = []
        past_h = self.h_init
        for train, test in cv.split(X):
            m = clone(self.estimator)
            m.fit(X.iloc[train], y.iloc[train])
            if self.vol_liquidity_factor is None:
                vlf = None
            else:
                vlf = np.squeeze(self.vol_liquidity_factor.values[test])
            current_h = m.predict(
                X.iloc[test], past_h=past_h, vol_liquidity_factor=vlf
            )
            _h += [current_h]
            past_h = current_h

        cols = X.columns
        idx = X.index[np.concatenate([test for _, test in cv.split(X)])]
        h_ = pd.DataFrame(np.concatenate(_h), index=idx, columns=cols)

        self.h_ = h_
        if self.return_pnl_component:
            self.pnl_ = compute_pnl_components(
                self.h_, ret, vol_liquidity_factor=self.vol_liquidity_factor
            )
        else:
            self.pnl_ = (
                h_.shift(1).mul(ret).sum(axis=1)[self.start_date : self.end_date]
            )
        return self
