"""
Portfolio strategy runner implementing:
  1. HMMStrategy: Wasserstein HMM + adaptive order selection + transaction-cost MVO
  2. KNNStrategy: KNN conditional moment estimator + transaction-cost MVO

Both strategies are strictly causal: the decision at time t uses only
information available through t-1 (lagged feature vectors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.hmm_model import (
    AdaptiveHMMSelector,
    fit_gaussian_hmm,
    get_predictive_state_probs,
    extract_hmm_components,
    K_MIN, K_MAX, VAL_SLICE_LENGTH, ORDER_SELECT_FREQ, COMPLEXITY_PENALTY,
    HMM_N_ITER,
)
from src.wasserstein import WassersteinTemplateTracker
from src.optimizer import solve_mvo, equal_weights
from src.knn_baseline import KNNMomentEstimator
from src.metrics import compute_effective_n

logger = logging.getLogger(__name__)

N_ASSETS = 5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    """Container for daily strategy outputs."""
    dates: list = field(default_factory=list)
    weights: list = field(default_factory=list)           # (T, N)
    returns: list = field(default_factory=list)           # (T,)
    turnovers: list = field(default_factory=list)         # (T,)
    n_eff: list = field(default_factory=list)             # (T,)
    # HMM-specific fields
    selected_k: list = field(default_factory=list)        # (T,)
    dominant_template: list = field(default_factory=list) # (T,)
    template_probs: list = field(default_factory=list)    # (T, G)

    def to_dataframes(self, asset_names: list[str]) -> dict[str, pd.DataFrame | pd.Series]:
        """Convert result lists to Pandas objects."""
        idx = pd.DatetimeIndex(self.dates)
        weights_df = pd.DataFrame(self.weights, index=idx, columns=asset_names)
        returns_s = pd.Series(self.returns, index=idx, name="portfolio_return")
        turnovers_s = pd.Series(self.turnovers, index=idx, name="turnover")
        n_eff_s = pd.Series(self.n_eff, index=idx, name="n_eff")
        out = {
            "weights": weights_df,
            "returns": returns_s,
            "turnovers": turnovers_s,
            "n_eff": n_eff_s,
        }
        if self.selected_k:
            out["selected_k"] = pd.Series(self.selected_k, index=idx, name="selected_k")
        if self.dominant_template:
            out["dominant_template"] = pd.Series(
                self.dominant_template, index=idx, name="dominant_template"
            )
        if self.template_probs and self.template_probs[0] is not None:
            n_t = len(self.template_probs[0])
            out["template_probs"] = pd.DataFrame(
                self.template_probs, index=idx,
                columns=[f"Template_{g}" for g in range(n_t)]
            )
        return out


# ---------------------------------------------------------------------------
# HMM Strategy
# ---------------------------------------------------------------------------

class HMMStrategy:
    """Expanding-window Wasserstein HMM + MVO portfolio strategy.

    Parameters
    ----------
    n_templates : int
        Number of persistent Wasserstein templates (G). Default: 6.
    eta : float
        Template exponential smoothing rate. Default: 0.05.
    gamma : float
        MVO risk aversion. Default: 1.0.
    tau : float
        MVO transaction-cost (L1 turnover) penalty. Default: 0.01.
    w_max : float
        Maximum per-asset weight. Default: 0.6.
    k_min, k_max : int
        Candidate regime counts for adaptive order selection.
    val_slice_length : int
        Validation slice length for order selection.
    select_freq : int
        Re-select order every `select_freq` trading days.
    complexity_penalty : float
        Linear penalty coefficient for adaptive K.
    random_state : int
        Global random seed.
    n_iter : int
        Maximum EM iterations for every HMM fit. Default: 10.
    max_train_size : int
        Cap on the rolling history passed to each HMM fit.  The paper
        describes a true expanding window, but fitting on 4 000+ obs per
        step is infeasible on a single CPU.  Setting max_train_size=1000
        caps history at ~4 years, which retains all regime structure while
        bounding per-fit cost.  Set to 0 for a true expanding window.
    """

    def __init__(
        self,
        n_templates: int = 6,
        eta: float = 0.05,
        gamma: float = 1.0,
        tau: float = 0.01,
        w_max: float = 0.6,
        k_min: int = K_MIN,
        k_max: int = K_MAX,
        val_slice_length: int = VAL_SLICE_LENGTH,
        select_freq: int = ORDER_SELECT_FREQ,
        complexity_penalty: float = COMPLEXITY_PENALTY,
        random_state: int = 42,
        n_assets: int = N_ASSETS,
        n_iter: int = HMM_N_ITER,
        max_train_size: int = 1000,
    ) -> None:
        self.n_templates = n_templates
        self.eta = eta
        self.gamma = gamma
        self.tau = tau
        self.w_max = w_max
        self.k_min = k_min
        self.k_max = k_max
        self.val_slice_length = val_slice_length
        self.select_freq = select_freq
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        self.n_assets = n_assets
        self.n_iter = n_iter
        self.max_train_size = max_train_size

        self.tracker = WassersteinTemplateTracker(
            n_templates=n_templates, eta=eta,
            feature_dim=3 * n_assets, n_assets=n_assets,
        )
        self.selector = AdaptiveHMMSelector(
            k_min=k_min, k_max=k_max,
            val_slice_length=val_slice_length,
            select_freq=select_freq,
            penalty=complexity_penalty,
            random_state=random_state,
            n_iter=n_iter,
        )

        self._current_k: int = k_min
        self._initialized_templates: bool = False

    def _initialize_templates(self, X_train: np.ndarray) -> None:
        """Fit initial HMM on training data to seed the template bank."""
        init_k = min(self.n_templates, self.k_max)
        model = fit_gaussian_hmm(X_train, init_k, random_state=self.random_state, n_iter=self.n_iter)
        if model is not None:
            means, covars = extract_hmm_components(model)
            self.tracker.initialize_from_hmm(means, covars)
            logger.info(
                "Templates initialized from K=%d HMM on %d training obs.",
                init_k, len(X_train),
            )
        else:
            logger.warning("Initial HMM failed; using zero/identity templates.")
        self._initialized_templates = True

    def run(
        self,
        lagged_features: pd.DataFrame,
        returns: pd.DataFrame,
        train_end_date: str,
    ) -> StrategyResult:
        """Run the strategy over the out-of-sample period.

        Parameters
        ----------
        lagged_features : pd.DataFrame
            Strictly causal feature matrix (t uses info through t-1).
        returns : pd.DataFrame
            Realized daily log-returns (same index as lagged_features).
        train_end_date : str
            Date string marking end of in-sample period (inclusive).
            OOS starts the next trading day.

        Returns
        -------
        StrategyResult
        """
        train_mask = lagged_features.index <= pd.Timestamp(train_end_date)
        oos_mask = ~train_mask

        X_all = lagged_features.values.astype(float)
        R_all = returns.values.astype(float)
        dates_all = lagged_features.index

        train_idx = np.where(train_mask)[0]
        oos_idx = np.where(oos_mask)[0]

        if len(train_idx) == 0 or len(oos_idx) == 0:
            raise ValueError("No in-sample or out-of-sample data found.")

        # Seed templates from training data
        self._initialize_templates(X_all[train_idx])

        result = StrategyResult()
        w_prev = equal_weights(self.n_assets)

        logger.info(
            "HMMStrategy OOS: %d days (%s → %s)",
            len(oos_idx),
            dates_all[oos_idx[0]].date(),
            dates_all[oos_idx[-1]].date(),
        )

        for step_i, t in enumerate(oos_idx):
            date = dates_all[t]
            # Expanding history: all data strictly before t
            # Expanding window capped at max_train_size for compute feasibility.
            # The paper specifies a true expanding window; max_train_size=1000
            # retains ~4 years of regime history while keeping each fit fast.
            start_t = max(0, t - self.max_train_size) if self.max_train_size > 0 else 0
            X_hist = X_all[start_t:t]  # shape (min(t, max_train_size), D)
            x_t = X_all[t]      # feature vector used for decision at date t
            r_t = R_all[t]      # realized return on date t

            if len(X_hist) < self.k_min + 10:
                w_t = w_prev.copy()
                result.dates.append(date)
                result.weights.append(w_t.copy())
                result.returns.append(float(r_t @ w_t))
                result.turnovers.append(0.0)
                result.n_eff.append(float(1.0 / np.sum(w_t ** 2)))
                result.selected_k.append(self._current_k)
                result.dominant_template.append(0)
                result.template_probs.append(
                    np.ones(self.n_templates) / self.n_templates
                )
                w_prev = w_t
                continue

            # --- Adaptive order selection (every select_freq steps) ---
            selected_k, _ = self.selector.fit_predict(X_hist, step_i, self._current_k)
            self._current_k = selected_k

            # --- Fit HMM on expanding history ---
            model = fit_gaussian_hmm(
                X_hist, self._current_k, random_state=self.random_state, n_iter=self.n_iter
            )
            if model is None:
                # Fallback: use previous weights
                w_t = w_prev.copy()
                result.dates.append(date)
                result.weights.append(w_t.copy())
                result.returns.append(float(r_t @ w_t))
                result.turnovers.append(0.0)
                result.n_eff.append(float(1.0 / np.sum(w_t ** 2)))
                result.selected_k.append(self._current_k)
                result.dominant_template.append(getattr(self.tracker, "dominant", 0))
                result.template_probs.append(
                    np.ones(self.n_templates) / self.n_templates
                )
                w_prev = w_t
                continue

            # --- One-step-ahead predictive state probabilities ---
            state_probs = get_predictive_state_probs(model, X_hist)

            # --- Extract HMM component parameters ---
            comp_means, comp_covars = extract_hmm_components(model)

            # --- Wasserstein template step ---
            template_probs, assignments, mu_ret, Sigma_ret = self.tracker.step(
                comp_means, comp_covars, state_probs
            )

            # --- MVO optimization ---
            # Annualise moments (daily → annual) so the return signal is large
            # enough relative to the tau L1-turnover penalty.  Both mu and Sigma
            # are multiplied by 252; the turnover term is unchanged (one-time
            # per-rebalancing cost).  The MVO weights are scale-invariant when
            # only the return/risk ratio changes, but the tau interaction now
            # correctly incentivises rebalancing only when the gain exceeds the
            # annual-equivalent transaction cost.
            _ANN = 252
            w_t = solve_mvo(
                mu=mu_ret * _ANN,
                Sigma=Sigma_ret * _ANN,
                w_prev=w_prev,
                gamma=self.gamma,
                tau=self.tau,
                w_max=self.w_max,
            )

            # --- Portfolio realized return ---
            port_ret = float(r_t @ w_t)
            turnover = 0.5 * float(np.sum(np.abs(w_t - w_prev)))
            n_eff = float(1.0 / np.sum(w_t ** 2))

            result.dates.append(date)
            result.weights.append(w_t.copy())
            result.returns.append(port_ret)
            result.turnovers.append(turnover)
            result.n_eff.append(n_eff)
            result.selected_k.append(self._current_k)
            result.dominant_template.append(int(self.tracker.dominant))
            result.template_probs.append(template_probs.copy())

            w_prev = w_t

            if step_i % 50 == 0:
                logger.info(
                    "HMM step %d/%d | date=%s | K=%d | dominant_template=%d | ret=%.4f",
                    step_i + 1, len(oos_idx), date.date(),
                    self._current_k, int(self.tracker.dominant), port_ret,
                )

        return result


# ---------------------------------------------------------------------------
# KNN Strategy
# ---------------------------------------------------------------------------

class KNNStrategy:
    """KNN conditional moment estimator + transaction-cost MVO strategy.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbours.
    gamma, tau, w_max : float
        MVO hyperparameters (same as HMMStrategy for fair comparison).
    n_assets : int
        Number of assets.
    """

    def __init__(
        self,
        k_neighbors: int = 30,
        gamma: float = 1.0,
        tau: float = 0.01,
        w_max: float = 0.6,
        n_assets: int = N_ASSETS,
        metric: str = "euclidean",
    ) -> None:
        self.k_neighbors = k_neighbors
        self.gamma = gamma
        self.tau = tau
        self.w_max = w_max
        self.n_assets = n_assets
        self.estimator = KNNMomentEstimator(k=k_neighbors, metric=metric)

    def run(
        self,
        lagged_features: pd.DataFrame,
        returns: pd.DataFrame,
        train_end_date: str,
    ) -> StrategyResult:
        """Run KNN strategy over out-of-sample period.

        Parameters
        ----------
        lagged_features : pd.DataFrame
            Strictly causal feature matrix.
        returns : pd.DataFrame
            Realized daily log-returns.
        train_end_date : str
            End of in-sample period.
        """
        train_mask = lagged_features.index <= pd.Timestamp(train_end_date)
        oos_mask = ~train_mask

        X_all = lagged_features.values.astype(float)
        R_all = returns.values.astype(float)
        dates_all = lagged_features.index

        train_idx = np.where(train_mask)[0]
        oos_idx = np.where(oos_mask)[0]

        # Seed KNN with in-sample history
        if len(train_idx) > self.k_neighbors + 5:
            self.estimator.fit_from_history(
                X_all[train_idx], R_all[train_idx]
            )

        result = StrategyResult()
        w_prev = equal_weights(self.n_assets)

        logger.info(
            "KNNStrategy OOS: %d days (%s → %s)",
            len(oos_idx),
            dates_all[oos_idx[0]].date(),
            dates_all[oos_idx[-1]].date(),
        )

        for step_i, t in enumerate(oos_idx):
            date = dates_all[t]
            x_t = X_all[t]   # feature vector (already lagged)
            r_t = R_all[t]   # realized return

            # Estimate conditional moments using expanding history
            moments = self.estimator.estimate_moments(x_t, n_assets=self.n_assets)

            if moments is None:
                w_t = w_prev.copy()
            else:
                mu_cond, Sigma_cond = moments
                _ANN = 252  # annualise daily moments (same convention as HMM)
                w_t = solve_mvo(
                    mu=mu_cond * _ANN,
                    Sigma=Sigma_cond * _ANN,
                    w_prev=w_prev,
                    gamma=self.gamma,
                    tau=self.tau,
                    w_max=self.w_max,
                )

            # Append current (lagged feature, return) to history for next step
            # Note: at date t, x_t is lagged feature, r_t is contemporaneous return
            self.estimator.update(X_all[t], R_all[t])

            port_ret = float(r_t @ w_t)
            turnover = 0.5 * float(np.sum(np.abs(w_t - w_prev)))
            n_eff = float(1.0 / np.sum(w_t ** 2))

            result.dates.append(date)
            result.weights.append(w_t.copy())
            result.returns.append(port_ret)
            result.turnovers.append(turnover)
            result.n_eff.append(n_eff)

            w_prev = w_t

            if step_i % 50 == 0:
                logger.info(
                    "KNN step %d/%d | date=%s | ret=%.4f",
                    step_i + 1, len(oos_idx), date.date(), port_ret,
                )

        return result
