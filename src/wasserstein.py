"""
Wasserstein-distance template-based regime tracking.

Provides:

* Closed-form squared 2-Wasserstein distance between Gaussian distributions.
* PSD matrix utilities (square root, projection).
* :class:`WassersteinTemplateTracker` — maintains a fixed bank of *G* persistent
  regime templates that are assigned, aggregated, and updated online as new HMM
  fits arrive.

The tracker decouples the number of HMM states (which can vary across refits)
from the number of templates *G*, enabling stable long-run tracking even when
the fitted order changes.
"""

import logging

import numpy as np
import scipy.linalg  # noqa: F401 – kept for callers that may use it directly

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSD matrix utilities
# ---------------------------------------------------------------------------

def psd_sqrt(M: np.ndarray) -> np.ndarray:
    """Compute the symmetric PSD square root of a symmetric positive semi-definite matrix.

    Uses an eigendecomposition, clips eigenvalues at ``1e-10`` to handle
    near-singular matrices, and symmetrizes the result to suppress floating-point
    asymmetry.

    Parameters
    ----------
    M:
        Symmetric PSD matrix of shape ``(D, D)``.

    Returns
    -------
    np.ndarray
        Shape ``(D, D)`` — the matrix square root ``M^{1/2}`` satisfying
        ``M^{1/2} @ M^{1/2} ≈ M``.
    """
    M_sym = (M + M.T) / 2.0
    vals, vecs = np.linalg.eigh(M_sym)
    vals = np.clip(vals, 1e-10, None)
    sqrt_diag = np.diag(np.sqrt(vals))
    result = vecs @ sqrt_diag @ vecs.T
    return (result + result.T) / 2.0


def psd_project(M: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone.

    Symmetrizes *M* then clips all negative eigenvalues to zero, returning the
    nearest (in Frobenius norm) PSD matrix.

    Parameters
    ----------
    M:
        Square matrix of shape ``(D, D)``.

    Returns
    -------
    np.ndarray
        Shape ``(D, D)`` — nearest symmetric PSD matrix.
    """
    M_sym = (M + M.T) / 2.0
    vals, vecs = np.linalg.eigh(M_sym)
    vals = np.clip(vals, 0.0, None)
    result = vecs @ np.diag(vals) @ vecs.T
    return (result + result.T) / 2.0


# ---------------------------------------------------------------------------
# Wasserstein-2 distance for Gaussians
# ---------------------------------------------------------------------------

def gaussian_w2_squared(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    r"""Compute the squared 2-Wasserstein distance between two Gaussians.

    The closed-form expression is::

        W2²(N(μ₁,Σ₁), N(μ₂,Σ₂))
            = ‖μ₁ − μ₂‖²
              + Tr(Σ₁ + Σ₂ − 2·(Σ₂^{1/2} Σ₁ Σ₂^{1/2})^{1/2})

    Numerical details:

    * ``Σ₂^{1/2}`` is computed via :func:`psd_sqrt` (eigenvalue clipping at ``1e-10``).
    * The inner matrix ``Σ₂^{1/2} Σ₁ Σ₂^{1/2}`` is symmetrized before its own
      square root is taken.
    * The trace term is clipped at zero to absorb tiny negative values caused by
      floating-point rounding.

    Parameters
    ----------
    mu1, mu2:
        Mean vectors of shape ``(D,)``.
    Sigma1, Sigma2:
        Covariance matrices of shape ``(D, D)``.

    Returns
    -------
    float
        Non-negative squared Wasserstein-2 distance.
    """
    # Mean contribution.
    mean_term = float(np.sum((mu1 - mu2) ** 2))

    # Covariance contribution.
    S2_sqrt = psd_sqrt(Sigma2)                   # Σ₂^{1/2}
    inner = S2_sqrt @ Sigma1 @ S2_sqrt           # Σ₂^{1/2} Σ₁ Σ₂^{1/2}
    inner_sym = (inner + inner.T) / 2.0          # symmetrize before sqrt
    M_sqrt = psd_sqrt(inner_sym)                 # (Σ₂^{1/2} Σ₁ Σ₂^{1/2})^{1/2}

    trace_term = float(np.trace(Sigma1) + np.trace(Sigma2) - 2.0 * np.trace(M_sqrt))
    trace_term = max(0.0, trace_term)            # clip tiny negatives

    return mean_term + trace_term


# ---------------------------------------------------------------------------
# Template tracker
# ---------------------------------------------------------------------------

class WassersteinTemplateTracker:
    """Persistent template bank for online Wasserstein-based regime tracking.

    Maintains *G* regime templates, each represented as a Gaussian
    ``(μ_g, Σ_g)``.  On every time step the current HMM components are matched
    to templates via minimum-W2 assignment, template probabilities are formed
    by aggregating predictive state weights, and templates are updated with an
    exponential moving average.

    Parameters
    ----------
    n_templates:
        Number of persistent templates ``G``.
    eta:
        Learning rate for the exponential template update (``0 < η ≤ 1``).
    feature_dim:
        Full feature dimension ``D`` (length of each component mean vector).
    n_assets:
        Number of return series ``N``; the first *N* features are treated as
        returns for moment extraction.
    """

    def __init__(
        self,
        n_templates: int = 6,
        eta: float = 0.05,
        feature_dim: int = 15,
        n_assets: int = 5,
    ) -> None:
        self.n_templates = n_templates
        self.eta = eta
        self.feature_dim = feature_dim
        self.n_assets = n_assets

        self.template_means: np.ndarray = np.zeros((n_templates, feature_dim))
        self.template_covars: np.ndarray = np.stack(
            [np.eye(feature_dim)] * n_templates, axis=0
        )
        self.initialized: bool = False
        self.dominant: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_from_hmm(
        self,
        model_means: np.ndarray,
        model_covars: np.ndarray,
    ) -> None:
        """Seed templates from the components of an initial HMM fit.

        If the model has fewer than *G* components the components are cycled
        (repeated round-robin) to fill all *G* template slots.  If the model
        has more than *G* components only the first *G* are used.

        Parameters
        ----------
        model_means:
            HMM component means, shape ``(K, D)``.
        model_covars:
            HMM component covariances, shape ``(K, D, D)``.
        """
        K = model_means.shape[0]
        for g in range(self.n_templates):
            k = g % K                                       # cycle if K < G
            self.template_means[g] = model_means[k].copy()
            self.template_covars[g] = model_covars[k].copy()

        self.initialized = True
        logger.info(
            "Templates initialized from HMM with K=%d into G=%d slots.", K, self.n_templates
        )

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign_components_to_templates(
        self,
        comp_means: np.ndarray,
        comp_covars: np.ndarray,
    ) -> np.ndarray:
        """Assign each HMM component to its nearest template in W2 distance.

        Parameters
        ----------
        comp_means:
            Component means, shape ``(K, D)``.
        comp_covars:
            Component covariances, shape ``(K, D, D)``.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(K,)`` with the index of the closest
            template for each component.
        """
        K = comp_means.shape[0]
        assignments = np.empty(K, dtype=int)

        for k in range(K):
            best_g = 0
            min_dist = np.inf
            for g in range(self.n_templates):
                dist = gaussian_w2_squared(
                    comp_means[k], comp_covars[k],
                    self.template_means[g], self.template_covars[g],
                )
                if dist < min_dist:
                    min_dist = dist
                    best_g = g
            assignments[k] = best_g

        logger.debug("Component-to-template assignments: %s", assignments.tolist())
        return assignments

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_template_probs(
        self,
        state_probs: np.ndarray,
        assignments: np.ndarray,
    ) -> np.ndarray:
        """Aggregate predictive state weights into template probabilities.

        Parameters
        ----------
        state_probs:
            Predictive state probabilities ``p(z_t | F_{t-1})``, shape ``(K,)``.
        assignments:
            Template index for each component, shape ``(K,)`` — output of
            :meth:`assign_components_to_templates`.

        Returns
        -------
        np.ndarray
            Shape ``(G,)`` — non-negative template probabilities summing to 1.
        """
        template_probs = np.zeros(self.n_templates)
        for k, g in enumerate(assignments):
            template_probs[g] += state_probs[k]

        total = template_probs.sum()
        if total > 0.0:
            template_probs /= total
        else:
            logger.warning("Zero total weight in template aggregation; using uniform probs.")
            template_probs[:] = 1.0 / self.n_templates

        return template_probs

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_templates(
        self,
        comp_means: np.ndarray,
        comp_covars: np.ndarray,
        state_probs: np.ndarray,
        assignments: np.ndarray,
    ) -> None:
        """Update template parameters with an exponential moving average.

        For each template *g* with at least one assigned component:

        1. Compute a weighted average ``(μ̄_g, Σ̄_g)`` using the normalised
           ``state_probs`` of the assigned components as weights.
        2. Apply the EMA update::

               μ_g  ← (1 − η)·μ_g  + η·μ̄_g
               Σ_g  ← (1 − η)·Σ_g  + η·Σ̄_g

        3. Project the updated covariance onto the PSD cone via
           :func:`psd_project`.

        Parameters
        ----------
        comp_means:
            Component means, shape ``(K, D)``.
        comp_covars:
            Component covariances, shape ``(K, D, D)``.
        state_probs:
            Predictive state probabilities, shape ``(K,)``.
        assignments:
            Template index for each component, shape ``(K,)``.
        """
        for g in range(self.n_templates):
            mask = assignments == g
            if not mask.any():
                logger.debug("Template g=%d has no assigned components; skipping update.", g)
                continue

            weights = state_probs[mask]
            w_sum = weights.sum()
            if w_sum <= 0.0:
                logger.debug("Template g=%d: zero total weight; skipping update.", g)
                continue
            weights = weights / w_sum                        # normalise within group

            mu_bar = np.einsum("k,kd->d", weights, comp_means[mask])
            Sigma_bar = np.einsum("k,kij->ij", weights, comp_covars[mask])

            self.template_means[g] = (1.0 - self.eta) * self.template_means[g] + self.eta * mu_bar
            Sigma_updated = (1.0 - self.eta) * self.template_covars[g] + self.eta * Sigma_bar
            self.template_covars[g] = psd_project(Sigma_updated)

    # ------------------------------------------------------------------
    # Return-space moments
    # ------------------------------------------------------------------

    def get_return_moments(
        self,
        template_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute return-space mixture mean and covariance for MVO.

        For each template *g* the return block is taken as the first *n_assets*
        features of ``template_means[g]`` and the top-left
        ``(n_assets, n_assets)`` block of ``template_covars[g]``.  The mixture
        moments are::

            μ_t    = Σ_g p_g · μ_g^{ret}
            Σ_t    = Σ_g p_g · Σ_g^{ret}

        ``Σ_t`` is projected to the PSD cone before being returned.

        Parameters
        ----------
        template_probs:
            Shape ``(G,)`` — template probability weights summing to 1.

        Returns
        -------
        mu_t : np.ndarray
            Shape ``(N,)`` — mixture return mean.
        Sigma_t : np.ndarray
            Shape ``(N, N)`` — mixture return covariance (PSD-projected).
        """
        N = self.n_assets
        mu_t = np.zeros(N)
        Sigma_t = np.zeros((N, N))

        for g in range(self.n_templates):
            p_g = template_probs[g]
            mu_t += p_g * self.template_means[g, :N]
            Sigma_t += p_g * self.template_covars[g, :N, :N]

        Sigma_t = psd_project(Sigma_t)
        return mu_t, Sigma_t

    # ------------------------------------------------------------------
    # Single daily step
    # ------------------------------------------------------------------

    def step(
        self,
        model_means: np.ndarray,
        model_covars: np.ndarray,
        state_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute a full daily template-tracking step.

        Pipeline:

        1. Assign HMM components to templates (minimum W2).
        2. Aggregate predictive state probabilities into template probabilities.
        3. Update templates with EMA.
        4. Record dominant template (``argmax`` of template probabilities).
        5. Extract return-space moments for MVO.

        Parameters
        ----------
        model_means:
            Current HMM component means, shape ``(K, D)``.
        model_covars:
            Current HMM component covariances, shape ``(K, D, D)``.
        state_probs:
            Predictive state probabilities ``p(z_t | F_{t-1})``, shape ``(K,)``.

        Returns
        -------
        template_probs : np.ndarray
            Shape ``(G,)`` — template probabilities.
        assignments : np.ndarray
            Shape ``(K,)`` — template index assigned to each HMM component.
        mu_ret : np.ndarray
            Shape ``(N,)`` — return-space mixture mean.
        Sigma_ret : np.ndarray
            Shape ``(N, N)`` — return-space mixture covariance.
        """
        assignments = self.assign_components_to_templates(model_means, model_covars)
        template_probs = self.aggregate_template_probs(state_probs, assignments)
        self.update_templates(model_means, model_covars, state_probs, assignments)

        self.dominant = int(np.argmax(template_probs))
        logger.debug(
            "Dominant template: g=%d (prob=%.4f).", self.dominant, template_probs[self.dominant]
        )

        mu_ret, Sigma_ret = self.get_return_moments(template_probs)
        return template_probs, assignments, mu_ret, Sigma_ret
