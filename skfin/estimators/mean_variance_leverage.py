"""Mean-variance optimization with leverage constraints (L1-norm penalty)."""

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator

from skfin.estimators.mean_variance import MeanVariance

logger = logging.getLogger(__name__)


def compute_holdings_with_leverage(
    pred: np.ndarray,
    V: np.ndarray,
    A: np.ndarray | None = None,
    risk_target: float | None = None,
    leverage_target: float | None = None,
) -> np.ndarray:
    """Solve convex optimization for holdings with optional leverage constraint.

    Maximizes expected return subject to risk and L1-norm (leverage) constraints.
    Uses CVXPY with the SCS solver.

    Args:
        pred: Predicted returns, shape (N,).
        V: Covariance matrix, shape (N, N).
        A: Constraint matrix for neutrality. None for unconstrained.
        risk_target: Maximum portfolio variance (quadratic constraint).
        leverage_target: Maximum sum of absolute holdings (L1 constraint).

    Returns:
        Optimal holdings array, shape (N,).
    """
    n = len(pred)
    h = cp.Variable(n)

    objective = cp.Maximize(h @ pred)

    constraints = []
    if A is not None:
        constraints.append(h.T @ A == 0)
    if risk_target is not None:
        constraints.append(cp.quad_form(h, V) <= risk_target)
    if leverage_target is not None:
        constraints.append(cp.norm(h, 1) <= leverage_target)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=1e-9, max_iters=10000, verbose=False)

    return h.value


@dataclass
class MeanVarianceWithLeverage(MeanVariance):
    """Mean-variance optimizer with an L1-norm leverage constraint."""

    leverage_target: float | None = None

    def __post_init__(self):
        self.holdings_kwargs = {
            "risk_target": self.risk_target,
            "leverage_target": self.leverage_target,
        }

    @staticmethod
    def compute_batch_holdings(
        pred: np.ndarray,
        V: np.ndarray,
        A: np.ndarray | None,
        risk_target: float | None = None,
        leverage_target: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute leverage-constrained holdings via convex optimization.

        Args:
            pred: Predicted returns, shape (N,) or (K, N).
            V: Covariance matrix, shape (N, N).
            A: Constraint matrix. None for unconstrained.
            risk_target: Maximum portfolio variance.
            leverage_target: Maximum L1-norm of holdings.

        Returns:
            Holdings array, shape (1, N).
        """
        pred = pred.squeeze()
        h = compute_holdings_with_leverage(
            pred, V, A, risk_target=risk_target, leverage_target=leverage_target
        )
        return h[np.newaxis, :]
