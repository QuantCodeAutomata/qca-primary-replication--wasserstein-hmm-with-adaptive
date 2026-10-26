"""
Gaussian HMM fitting and adaptive order selection for regime detection.

Provides utilities for fitting Hidden Markov Models with Gaussian emissions,
selecting the number of hidden states via predictive log-likelihood with a
complexity penalty, and extracting predictive state distributions for downstream
portfolio optimisation.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler  # noqa: F401 – available for callers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_MIN: int = 2
K_MAX: int = 8
COMPLEXITY_PENALTY: float = 0.5
VAL_SLICE_LENGTH: int = 60
ORDER_SELECT_FREQ: int = 5
HMM_N_ITER: int = 30  # default EM iterations (reduced from 100 for speed)


# ---------------------------------------------------------------------------
# Core HMM fitting
# ---------------------------------------------------------------------------

def fit_gaussian_hmm(
    X: np.ndarray,
    n_components: int,
    random_state: int = 42,
    n_iter: int = 30,
    tol: float = 1e-3,
) -> GaussianHMM | None:
    """Fit a Gaussian HMM with *n_components* hidden states to *X*.

    Parameters
    ----------
    X:
        Observation matrix of shape ``(T, D)``.
    n_components:
        Number of hidden states.
    random_state:
        Seed for the random number generator, ensuring reproducibility.
    n_iter:
        Maximum number of EM iterations.
    tol:
        Convergence tolerance for the EM algorithm.

    Returns
    -------
    GaussianHMM | None
        Fitted model, or ``None`` when fitting fails for any reason.
    """
    try:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            min_covar=1e-4,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        logger.debug("Fitted GaussianHMM with K=%d on %d observations.", n_components, len(X))
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("HMM fitting failed for K=%d: %s", n_components, exc)
        return None


# ---------------------------------------------------------------------------
# Predictive log-likelihood
# ---------------------------------------------------------------------------

def compute_predictive_loglik(model: GaussianHMM, X_val: np.ndarray) -> float:
    """Compute one-step-ahead predictive log-likelihood over a validation slice.

    At each step *s* the predictive distribution ``p(x_s | x_{<s})`` is
    obtained by:

    1. Propagating the filtered distribution through the transition matrix.
    2. Forming the Gaussian mixture predictive density.
    3. Updating filtered probabilities with a Bayes step.

    Log-sum-exp is used throughout for numerical stability.

    Parameters
    ----------
    model:
        A fitted :class:`~hmmlearn.hmm.GaussianHMM`.
    X_val:
        Validation observations of shape ``(T_val, D)``.

    Returns
    -------
    float
        Total predictive log-likelihood summed over all validation steps.
    """
    K = model.n_components
    transmat = model.transmat_          # (K, K)
    means = model.means_                # (K, D)
    covars = model.covars_              # (K, D, D)

    # Initialise with the prior start distribution.
    filtered_probs = model.startprob_.copy()  # (K,)

    total_ll = 0.0

    for s in range(len(X_val)):
        x_s = X_val[s]

        # -- Predict: one-step-ahead predictive state distribution ----------
        pred_probs = filtered_probs @ transmat  # (K,)

        # -- Emission log-probabilities under each state --------------------
        # Ridge-regularise each covariance to ensure PSD before logpdf
        ridge = 1e-6 * np.eye(covars.shape[-1])
        log_emissions = np.array(
            [
                multivariate_normal.logpdf(
                    x_s, mean=means[k], cov=covars[k] + ridge, allow_singular=True
                )
                for k in range(K)
            ]
        )  # (K,)

        # -- Predictive log-likelihood: log sum_k p(z=k) * p(x_s | z=k) ----
        log_pred_probs = np.log(np.clip(pred_probs, 1e-300, None))
        log_joint = log_pred_probs + log_emissions            # (K,)
        log_px = logsumexp(log_joint)
        total_ll += float(log_px)

        # -- Bayes update: filtered ∝ pred_probs * emission -----------------
        log_norm = logsumexp(log_joint)
        filtered_probs = np.exp(log_joint - log_norm)

    logger.debug("Predictive log-likelihood over %d steps: %.4f", len(X_val), total_ll)
    return total_ll


# ---------------------------------------------------------------------------
# Order selection
# ---------------------------------------------------------------------------

def select_hmm_order(
    X_train: np.ndarray,
    X_val: np.ndarray,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    penalty: float = COMPLEXITY_PENALTY,
    random_state: int = 42,
    n_iter: int = HMM_N_ITER,
) -> tuple[int, dict]:
    """Select the best number of HMM states via penalised predictive log-likelihood.

    For each *K* in ``[k_min, k_max]`` a model is fitted on *X_train* and
    evaluated on *X_val*.  The penalised score is::

        penalised_score(K) = PredLL(K) - penalty * K

    Parameters
    ----------
    X_train:
        Training observations of shape ``(T_train, D)``.
    X_val:
        Validation observations of shape ``(T_val, D)``.
    k_min, k_max:
        Inclusive range of candidate state counts.
    penalty:
        Per-state complexity penalty applied to the predictive log-likelihood.
    random_state:
        RNG seed forwarded to :func:`fit_gaussian_hmm`.
    n_iter:
        Maximum EM iterations forwarded to :func:`fit_gaussian_hmm`.

    Returns
    -------
    best_K : int
        State count with the highest penalised score.  Falls back to *k_min*
        when all fits fail.
    scores : dict[int, tuple[float, float]]
        Mapping ``K -> (pred_ll, penalised_score)`` for successfully fitted
        models.
    """
    scores: dict[int, tuple[float, float]] = {}
    best_K = k_min
    best_score = -np.inf

    for K in range(k_min, k_max + 1):
        model = fit_gaussian_hmm(X_train, K, random_state=random_state, n_iter=n_iter)
        if model is None:
            logger.warning("Skipping K=%d — fitting failed.", K)
            continue

        pred_ll = compute_predictive_loglik(model, X_val)
        penalised = pred_ll - penalty * K
        scores[K] = (pred_ll, penalised)
        logger.debug("K=%d  PredLL=%.4f  Penalised=%.4f", K, pred_ll, penalised)

        if penalised > best_score:
            best_score = penalised
            best_K = K

    if not scores:
        logger.warning("All HMM fits failed; falling back to K=%d.", k_min)
        best_K = k_min

    logger.info("Selected K=%d (penalised score=%.4f).", best_K, best_score)
    return best_K, scores


# ---------------------------------------------------------------------------
# Adaptive selector
# ---------------------------------------------------------------------------

class AdaptiveHMMSelector:
    """Periodically re-selects the HMM order using an expanding history window.

    The selector avoids redundant computation by only running order selection
    on time steps where ``t % select_freq == 0``.  On all other steps the
    previously chosen order is returned unchanged.

    Parameters
    ----------
    k_min, k_max:
        Inclusive range of candidate state counts.
    val_slice_length:
        Number of most-recent observations reserved for validation.
    select_freq:
        Re-selection period in time steps.
    penalty:
        Per-state complexity penalty forwarded to :func:`select_hmm_order`.
    random_state:
        RNG seed for reproducibility.
    n_iter:
        Maximum EM iterations forwarded to :func:`fit_gaussian_hmm` via
        :func:`select_hmm_order`.
    """

    def __init__(
        self,
        k_min: int = K_MIN,
        k_max: int = K_MAX,
        val_slice_length: int = VAL_SLICE_LENGTH,
        select_freq: int = ORDER_SELECT_FREQ,
        penalty: float = COMPLEXITY_PENALTY,
        random_state: int = 42,
        n_iter: int = HMM_N_ITER,
    ) -> None:
        self.k_min = k_min
        self.k_max = k_max
        self.val_slice_length = val_slice_length
        self.select_freq = select_freq
        self.penalty = penalty
        self.random_state = random_state
        self.n_iter = n_iter

    def fit_predict(
        self,
        X_history: np.ndarray,
        t: int,
        current_K: int,
    ) -> tuple[int, dict]:
        """Return the best HMM order given the expanding history up to *t-1*.

        Parameters
        ----------
        X_history:
            All observations available before time *t*, shape ``(t, D)``.
        t:
            Current time index (0-based).  Order selection runs only when
            ``t % select_freq == 0``.
        current_K:
            The order currently in use, returned unchanged when selection is
            skipped.

        Returns
        -------
        selected_K : int
            Chosen number of hidden states.
        scores : dict[int, tuple[float, float]]
            Score dictionary from :func:`select_hmm_order`, empty when
            selection is skipped.
        """
        if t % self.select_freq != 0:
            logger.debug("t=%d: skipping order selection (not a multiple of %d).", t, self.select_freq)
            return current_K, {}

        # Require enough data for a non-trivial training split.
        min_required = self.val_slice_length + self.k_min + 1
        if len(X_history) <= min_required:
            logger.debug(
                "t=%d: insufficient history (%d <= %d); keeping K=%d.",
                t, len(X_history), min_required, current_K,
            )
            return current_K, {}

        X_train = X_history[: -self.val_slice_length]
        X_val = X_history[-self.val_slice_length :]

        logger.info(
            "t=%d: running order selection — train=%d, val=%d.",
            t, len(X_train), len(X_val),
        )
        return select_hmm_order(
            X_train,
            X_val,
            k_min=self.k_min,
            k_max=self.k_max,
            penalty=self.penalty,
            random_state=self.random_state,
            n_iter=self.n_iter,
        )


# ---------------------------------------------------------------------------
# Predictive state probabilities
# ---------------------------------------------------------------------------

def get_predictive_state_probs(
    model: GaussianHMM,
    X_history: np.ndarray,
) -> np.ndarray:
    """Return one-step-ahead predictive state probabilities ``p(z_t | F_{t-1})``.

    The filtered distribution at the last observed step is obtained via
    :meth:`~hmmlearn.hmm.GaussianHMM.predict_proba`, then propagated forward
    through the transition matrix.

    Parameters
    ----------
    model:
        A fitted :class:`~hmmlearn.hmm.GaussianHMM`.
    X_history:
        Past observations of shape ``(T, D)``.  Must have at least one row.

    Returns
    -------
    np.ndarray
        Shape ``(n_components,)``, non-negative and summing to 1.
    """
    filtered_probs = model.predict_proba(X_history)[-1]   # (K,)
    pred_probs = filtered_probs @ model.transmat_          # (K,)

    # Renormalise to guard against floating-point drift.
    total = pred_probs.sum()
    if total > 0.0:
        pred_probs = pred_probs / total
    else:
        pred_probs = np.ones(model.n_components) / model.n_components

    return pred_probs.astype(float)


# ---------------------------------------------------------------------------
# Component extraction
# ---------------------------------------------------------------------------

def extract_hmm_components(
    model: GaussianHMM,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and symmetrize the emission parameters of a fitted HMM.

    Parameters
    ----------
    model:
        A fitted :class:`~hmmlearn.hmm.GaussianHMM`.

    Returns
    -------
    means : np.ndarray
        Shape ``(K, D)`` — component mean vectors.
    covars : np.ndarray
        Shape ``(K, D, D)`` — symmetrized component covariance matrices.
    """
    means = model.means_.copy()                # (K, D)
    covars_raw = model.covars_.copy()          # (K, D, D)

    # Symmetrize each covariance matrix to remove floating-point asymmetry.
    covars = (covars_raw + covars_raw.transpose(0, 2, 1)) / 2.0

    return means, covars
