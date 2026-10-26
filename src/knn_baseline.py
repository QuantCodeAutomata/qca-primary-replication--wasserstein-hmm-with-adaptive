"""KNN conditional moment estimator for expected returns and covariance."""

import logging

import numpy as np
import pandas as pd  # noqa: F401 — kept for downstream type-compatibility
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

K_NEIGHBORS: int = 30


class KNNMomentEstimator:
    """Estimate conditional return moments via k-nearest-neighbour regression.

    For a query feature vector ``x``, the estimator finds the ``k`` most
    similar historical feature observations, collects their contemporaneous
    asset returns, and computes a conditional mean and covariance over that
    neighbourhood.

    Parameters
    ----------
    k:
        Number of nearest neighbours to use.
    metric:
        Distance metric passed to :class:`sklearn.neighbors.NearestNeighbors`.
    """

    def __init__(self, k: int = K_NEIGHBORS, metric: str = "euclidean") -> None:
        self.k = k
        self.metric = metric
        self.scaler = StandardScaler()
        self.nn = NearestNeighbors(n_neighbors=k, metric=metric)
        self.X_history: np.ndarray | None = None  # lagged features history
        self.R_history: np.ndarray | None = None  # corresponding returns

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def update(self, X_new: np.ndarray, R_new: np.ndarray) -> None:
        """Append a new (feature, return) observation pair to the history.

        Parameters
        ----------
        X_new:
            Lagged feature vector of shape ``(D,)`` or ``(1, D)``.
        R_new:
            Asset return vector of shape ``(N,)`` or ``(1, N)``.
        """
        X_new = np.atleast_2d(X_new)  # → (1, D)
        R_new = np.atleast_2d(R_new)  # → (1, N)

        if self.X_history is None:
            self.X_history = X_new
            self.R_history = R_new
        else:
            self.X_history = np.vstack([self.X_history, X_new])
            self.R_history = np.vstack([self.R_history, R_new])

    def fit_from_history(self, X_hist: np.ndarray, R_hist: np.ndarray) -> None:
        """Initialise the estimator from pre-built history arrays.

        Parameters
        ----------
        X_hist:
            Feature history of shape ``(T, D)``.
        R_hist:
            Return history of shape ``(T, N)``.
        """
        self.X_history = X_hist
        self.R_history = R_hist

    # ------------------------------------------------------------------
    # Moment estimation
    # ------------------------------------------------------------------

    def estimate_moments(
        self,
        x_query: np.ndarray,
        n_assets: int = 5,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimate conditional expected return and covariance.

        Uses the ``k`` nearest historical feature observations to construct
        a neighbourhood of returns, then computes the conditional mean and
        a Ledoit-Wolf shrinkage covariance over that neighbourhood.

        Parameters
        ----------
        x_query:
            Query feature vector of shape ``(D,)`` or ``(1, D)``.
        n_assets:
            Number of assets; used only as the size of the fallback identity
            covariance when ``k < 2``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] | None
            ``(mu_cond, Sigma_cond)`` where ``mu_cond`` has shape ``(N,)``
            and ``Sigma_cond`` has shape ``(N, N)``, or ``None`` when the
            history contains fewer than ``k + 5`` observations.
        """
        if self.X_history is None or self.R_history is None:
            logger.debug("No history available; returning None.")
            return None

        T = self.X_history.shape[0]
        min_samples = self.k + 5
        if T < min_samples:
            logger.debug(
                "Insufficient history: %d samples (need %d); returning None.",
                T,
                min_samples,
            )
            return None

        # 1. Fit scaler on training-window feature history only.
        self.scaler.fit(self.X_history)

        # 2. Scale feature history and query point with the same scaler.
        X_scaled = self.scaler.transform(self.X_history)
        x_q = np.atleast_2d(x_query)
        x_q_scaled = self.scaler.transform(x_q)  # shape (1, D)

        # 3. Fit NearestNeighbors on scaled feature history.
        self.nn.fit(X_scaled)

        # 4. Find k nearest neighbours of the scaled query.
        _, indices = self.nn.kneighbors(x_q_scaled)  # indices shape (1, k)
        neighbor_indices = indices[0]               # shape (k,)

        # 5. Collect neighbour returns.
        R_neighbors = self.R_history[neighbor_indices]  # shape (k, N)

        # 6. Conditional mean.
        mu_cond: np.ndarray = R_neighbors.mean(axis=0)

        # 7. Conditional covariance.
        if self.k >= 2:
            lw = LedoitWolf()
            lw.fit(R_neighbors)
            Sigma_cond: np.ndarray = lw.covariance_
        else:
            Sigma_cond = np.eye(n_assets) * 1e-4

        # 8. Return conditional moments.
        return mu_cond, Sigma_cond
