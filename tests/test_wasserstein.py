"""Tests for src/wasserstein.py — Wasserstein distance and template tracking."""

import numpy as np
import pytest

from src.wasserstein import (
    gaussian_w2_squared,
    psd_project,
    psd_sqrt,
    WassersteinTemplateTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_psd(D: int, seed: int = 0) -> np.ndarray:
    """Return a random PSD matrix of size (D, D)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D))
    return (A @ A.T) / D + np.eye(D) * 0.1


def _random_mean(D: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(D)


# ---------------------------------------------------------------------------
# psd_sqrt
# ---------------------------------------------------------------------------

class TestPsdSqrt:
    def test_sqrt_squared_equals_original(self):
        Sigma = _random_psd(5)
        S = psd_sqrt(Sigma)
        reconstructed = S @ S
        np.testing.assert_allclose(reconstructed, Sigma, atol=1e-8)

    def test_output_is_symmetric(self):
        Sigma = _random_psd(6)
        S = psd_sqrt(Sigma)
        np.testing.assert_allclose(S, S.T, atol=1e-12)

    def test_output_is_psd(self):
        Sigma = _random_psd(4)
        S = psd_sqrt(Sigma)
        eigvals = np.linalg.eigvalsh(S)
        assert (eigvals >= -1e-10).all()

    def test_identity_sqrt_is_identity(self):
        I = np.eye(5)
        S = psd_sqrt(I)
        np.testing.assert_allclose(S, I, atol=1e-10)

    def test_zero_matrix(self):
        # psd_sqrt applies an internal ridge (~1e-5), so sqrt(Z) ≈ diag(1e-5).
        Z = np.zeros((4, 4))
        S = psd_sqrt(Z)
        # Result must be near-zero PSD (within ridge magnitude)
        np.testing.assert_allclose(S, Z, atol=1e-4)


# ---------------------------------------------------------------------------
# psd_project
# ---------------------------------------------------------------------------

class TestPsdProject:
    def test_psd_matrix_unchanged(self):
        Sigma = _random_psd(5)
        projected = psd_project(Sigma)
        np.testing.assert_allclose(projected, Sigma, atol=1e-8)

    def test_output_is_psd(self):
        # Introduce negative eigenvalue
        M = _random_psd(4)
        # perturb to make slightly indefinite
        M[0, 0] -= 2.0
        proj = psd_project(M)
        eigvals = np.linalg.eigvalsh(proj)
        assert (eigvals >= -1e-10).all()

    def test_output_is_symmetric(self):
        M = np.arange(16, dtype=float).reshape(4, 4)
        M = (M + M.T) / 2
        proj = psd_project(M)
        np.testing.assert_allclose(proj, proj.T, atol=1e-12)

    def test_idempotent(self):
        M = _random_psd(5)
        p1 = psd_project(M)
        p2 = psd_project(p1)
        np.testing.assert_allclose(p1, p2, atol=1e-10)


# ---------------------------------------------------------------------------
# gaussian_w2_squared
# ---------------------------------------------------------------------------

class TestGaussianW2Squared:
    def test_same_distribution_gives_zero(self):
        mu = _random_mean(5)
        Sigma = _random_psd(5)
        w2 = gaussian_w2_squared(mu, Sigma, mu, Sigma)
        assert abs(w2) < 1e-8

    def test_non_negative(self):
        for seed in range(5):
            mu1, mu2 = _random_mean(5, seed), _random_mean(5, seed + 10)
            S1, S2 = _random_psd(5, seed), _random_psd(5, seed + 10)
            w2 = gaussian_w2_squared(mu1, S1, mu2, S2)
            assert w2 >= -1e-9, f"W2^2 is negative: {w2}"

    def test_mean_shift_contributes_squared_distance(self):
        Sigma = np.eye(4)
        mu1 = np.zeros(4)
        delta = np.array([1.0, 0.0, 0.0, 0.0])
        mu2 = mu1 + delta
        w2 = gaussian_w2_squared(mu1, Sigma, mu2, Sigma)
        # With identical covariances, covariance term = Tr(2Σ - 2Σ) = 0
        # So W2^2 = ||mu1-mu2||^2 = 1
        np.testing.assert_allclose(w2, 1.0, atol=1e-8)

    def test_symmetric_wrt_argument_order(self):
        mu1, mu2 = _random_mean(5, 1), _random_mean(5, 2)
        S1, S2 = _random_psd(5, 1), _random_psd(5, 2)
        w12 = gaussian_w2_squared(mu1, S1, mu2, S2)
        w21 = gaussian_w2_squared(mu2, S2, mu1, S1)
        np.testing.assert_allclose(w12, w21, atol=1e-6)


# ---------------------------------------------------------------------------
# WassersteinTemplateTracker
# ---------------------------------------------------------------------------

class TestWassersteinTemplateTracker:
    def _make_tracker(self, G: int = 4, D: int = 10, N: int = 3) -> WassersteinTemplateTracker:
        return WassersteinTemplateTracker(
            n_templates=G, eta=0.1, feature_dim=D, n_assets=N
        )

    def _make_model_components(
        self, K: int = 3, D: int = 10, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        means = rng.standard_normal((K, D))
        covars = np.stack([_random_psd(D, seed + k) for k in range(K)])
        return means, covars

    def _make_state_probs(self, K: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        p = rng.dirichlet(np.ones(K))
        return p

    # --- initialize_from_hmm ---

    def test_initialize_sets_initialized_flag(self):
        tracker = self._make_tracker()
        means, covars = self._make_model_components(K=4, D=10)
        tracker.initialize_from_hmm(means, covars)
        assert tracker.initialized

    def test_initialize_from_fewer_components_cycles(self):
        G, D = 6, 10
        tracker = WassersteinTemplateTracker(n_templates=G, feature_dim=D, n_assets=3)
        means, covars = self._make_model_components(K=3, D=D)  # K < G
        tracker.initialize_from_hmm(means, covars)
        assert tracker.template_means.shape == (G, D)

    # --- assign_components_to_templates ---

    def test_assignments_in_valid_range(self):
        G, D, K = 4, 10, 3
        tracker = self._make_tracker(G=G, D=D)
        means, covars = self._make_model_components(K=K, D=D)
        tracker.initialize_from_hmm(means, covars)
        assignments = tracker.assign_components_to_templates(means, covars)
        assert assignments.shape == (K,)
        assert (assignments >= 0).all()
        assert (assignments < G).all()

    # --- aggregate_template_probs ---

    def test_template_probs_sum_to_one(self):
        tracker = self._make_tracker(G=4, D=10, N=3)
        means, covars = self._make_model_components(K=3, D=10)
        tracker.initialize_from_hmm(means, covars)
        assignments = tracker.assign_components_to_templates(means, covars)
        state_probs = self._make_state_probs(3)
        tp = tracker.aggregate_template_probs(state_probs, assignments)
        np.testing.assert_allclose(tp.sum(), 1.0, atol=1e-8)

    def test_template_probs_non_negative(self):
        tracker = self._make_tracker(G=4, D=10, N=3)
        means, covars = self._make_model_components(K=3, D=10)
        tracker.initialize_from_hmm(means, covars)
        assignments = tracker.assign_components_to_templates(means, covars)
        state_probs = self._make_state_probs(3)
        tp = tracker.aggregate_template_probs(state_probs, assignments)
        assert (tp >= 0).all()

    # --- get_return_moments ---

    def test_return_moments_shape(self):
        G, D, N = 4, 10, 3
        tracker = self._make_tracker(G=G, D=D, N=N)
        means, covars = self._make_model_components(K=3, D=D)
        tracker.initialize_from_hmm(means, covars)
        tp = np.ones(G) / G
        mu, Sigma = tracker.get_return_moments(tp)
        assert mu.shape == (N,)
        assert Sigma.shape == (N, N)

    def test_return_covariance_is_psd(self):
        G, D, N = 4, 10, 3
        tracker = self._make_tracker(G=G, D=D, N=N)
        means, covars = self._make_model_components(K=3, D=D)
        tracker.initialize_from_hmm(means, covars)
        tp = np.ones(G) / G
        _, Sigma = tracker.get_return_moments(tp)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert (eigvals >= -1e-9).all()

    # --- step (full pipeline) ---

    def test_step_returns_correct_shapes(self):
        G, D, N, K = 4, 10, 3, 3
        tracker = self._make_tracker(G=G, D=D, N=N)
        means, covars = self._make_model_components(K=K, D=D)
        tracker.initialize_from_hmm(means, covars)
        state_probs = self._make_state_probs(K)
        tp, asn, mu, Sigma = tracker.step(means, covars, state_probs)
        assert tp.shape == (G,)
        assert asn.shape == (K,)
        assert mu.shape == (N,)
        assert Sigma.shape == (N, N)

    def test_step_template_probs_sum_to_one(self):
        G, D, N, K = 4, 10, 3, 3
        tracker = self._make_tracker(G=G, D=D, N=N)
        means, covars = self._make_model_components(K=K, D=D)
        tracker.initialize_from_hmm(means, covars)
        tp, *_ = tracker.step(means, covars, self._make_state_probs(K))
        np.testing.assert_allclose(tp.sum(), 1.0, atol=1e-8)

    def test_dominant_is_valid_template_index(self):
        G, D, N, K = 4, 10, 3, 3
        tracker = self._make_tracker(G=G, D=D, N=N)
        means, covars = self._make_model_components(K=K, D=D)
        tracker.initialize_from_hmm(means, covars)
        tracker.step(means, covars, self._make_state_probs(K))
        assert 0 <= tracker.dominant < G

    def test_repeated_steps_dont_crash(self):
        G, D, N = 6, 15, 5
        tracker = WassersteinTemplateTracker(n_templates=G, eta=0.05, feature_dim=D, n_assets=N)
        for step_seed in range(10):
            K = np.random.randint(2, 5)
            means, covars = self._make_model_components(K=K, D=D, seed=step_seed)
            tracker.initialize_from_hmm(means, covars)
            state_probs = self._make_state_probs(K, seed=step_seed)
            tracker.step(means, covars, state_probs)
        assert tracker.initialized
