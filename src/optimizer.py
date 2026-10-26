"""Transaction-cost-aware mean-variance portfolio optimizer using cvxpy."""

import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)

GAMMA: float = 1.0
TAU: float = 0.01
W_MAX: float = 0.6
N_ASSETS: int = 5


def solve_mvo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    gamma: float = GAMMA,
    tau: float = TAU,
    w_max: float = W_MAX,
) -> np.ndarray:
    """Solve a transaction-cost-aware mean-variance optimization problem.

    Maximizes the risk-adjusted return net of L1 turnover costs:

        maximize  mu' w - gamma * w' Sigma w - tau * ||w - w_prev||_1
        subject to  sum(w) = 1,  0 <= w <= w_max

    Parameters
    ----------
    mu:
        Expected return vector of shape (N,).
    Sigma:
        Covariance matrix of shape (N, N). Must be symmetric positive
        semidefinite.
    w_prev:
        Portfolio weights from the previous period, shape (N,).
    gamma:
        Risk-aversion coefficient. Higher values penalize variance more.
    tau:
        Transaction cost coefficient for L1 turnover penalty.
    w_max:
        Maximum weight allowed for any single asset.

    Returns
    -------
    np.ndarray
        Optimal portfolio weights of shape (N,), summing to 1 with all
        non-negative entries. Falls back to ``w_prev`` (normalized) when
        the solver fails.
    """
    N = len(mu)

    if w_max < 1.0 / N:
        logger.warning(
            "w_max=%.4f is less than 1/N=%.4f; problem may be infeasible.",
            w_max,
            1.0 / N,
        )

    w = cp.Variable(N)

    risk = cp.quad_form(w, cp.psd_wrap(Sigma))
    turnover = cp.norm1(w - w_prev)
    objective = cp.Maximize(mu @ w - gamma * risk - tau * turnover)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max,
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve()
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
            raise ValueError(f"Solver returned status '{problem.status}'.")
        w_opt = np.array(w.value, dtype=float)
    except Exception as exc:
        logger.warning("MVO solver failed (%s); returning previous weights.", exc)
        w_opt = w_prev.copy()

    # Clip small negatives introduced by numerical noise, then re-normalize.
    w_opt = np.clip(w_opt, 0.0, None)
    total = w_opt.sum()
    if total <= 0.0:
        logger.warning("All weights non-positive after clip; falling back to equal weights.")
        w_opt = np.ones(N) / N
    else:
        w_opt = w_opt / total

    return w_opt


def equal_weights(n: int = N_ASSETS) -> np.ndarray:
    """Return an equally weighted portfolio.

    Parameters
    ----------
    n:
        Number of assets.

    Returns
    -------
    np.ndarray
        Array of shape (n,) with every element equal to ``1/n``.
    """
    return np.ones(n) / n
