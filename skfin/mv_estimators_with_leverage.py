import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from typing import Optional
from skfin.mv_estimators import MeanVariance

def compute_holdings_with_leverage(
    pred, V, A=None, risk_target=None, leverage_target=None
):
    """
    Optimize the array h to maximize np.sum(h * pred) under specified constraints.

    Parameters:
    - pred (numpy.array): A 1D array representing the predicted returns.
    - V (numpy.array): A 2D covariance matrix of returns.
    - leverage_target (float): The desired leverage, defined as the sum of absolute values of h.

    Returns:
    - numpy.array: The optimal array h that maximizes the objective under the constraints.
    """
    n = len(pred)

    # Define the optimization variable
    h = cp.Variable(n)

    # Objective function: maximize the sum of element-wise multiplication of h and pred
    objective = cp.Maximize(cp.sum(h @ pred))

    # Constraints
    constraints = []
    if A is not None:
        constraints += [h.T @ A == 0]
    if risk_target is not None:
        constraints += [
            cp.quad_form(h, V) <= risk_target
        ]  # Ensures the quadratic form (h^T V h) is 1
    if leverage_target is not None:
        constraints += [
            cp.norm(h, 1) <= leverage_target
        ]  # Ensures the sum of absolute values of h equals leverage_target

    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, eps=1e-9, max_iters=10000, verbose=False)
    problem.solve()

    # Retrieve the optimal value for h
    h_value = h.value

    return h_value


@dataclass
class MeanVarianceWithLeverage(MeanVariance):
    """
    Mean-variance optimization estimator with leverage constraints.

    Attributes:
        leverage_target (Optional[float]): Leverage target for the portfolio.
    """

    leverage_target: Optional[float] = None

    def __post_init__(self):
        """
        Post-initialization to update holdings keyword arguments.
        """
        self.holdings_kwargs = {
            "risk_target": self.risk_target,
            "leverage_target": self.leverage_target,
        }

    @staticmethod
    def compute_batch_holdings(pred, V, A, **kwargs):
        """
        Compute portfolio holdings considering leverage.

        Parameters:
            pred (np.ndarray): Predicted returns (squeezed for computation).
            V (np.ndarray): Covariance matrix.
            A (np.ndarray): Constraint matrix.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Portfolio holdings considering leverage.
        """
        pred = pred.squeeze()  # Squeeze prediction array for computation
        h = compute_holdings_with_leverage(pred, V, A, **kwargs)
        return h[:, np.newaxis].T
