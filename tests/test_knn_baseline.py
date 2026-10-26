"""Tests for src/knn_baseline.py — KNN conditional moment estimator."""

import numpy as np
import pytest

from src.knn_baseline import KNNMomentEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(
    T: int = 100, D: int = 15, N: int = 5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, D))
    R = rng.standard_normal((T, N)) * 0.01
    return X, R


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKNNMomentEstimator:
    def test_fit_from_history_stores_data(self):
        X, R = _make_history(80)
        est = KNNMomentEstimator(k=5)
        est.fit_from_history(X, R)
        assert est.X_history is not None
        assert est.R_history is not None
        assert est.X_history.shape == X.shape
        assert est.R_history.shape == R.shape

    def test_estimate_moments_returns_tuple(self):
        X, R = _make_history(100)
        est = KNNMomentEstimator(k=10)
        est.fit_from_history(X, R)
        result = est.estimate_moments(X[0], n_assets=5)
        assert result is not None
        mu, Sigma = result
        assert mu.shape == (5,)
        assert Sigma.shape == (5, 5)

    def test_mu_is_finite(self):
        X, R = _make_history(100, N=5)
        est = KNNMomentEstimator(k=10)
        est.fit_from_history(X, R)
        result = est.estimate_moments(X[0])
        assert result is not None
        mu, _ = result
        assert np.all(np.isfinite(mu))

    def test_sigma_is_psd(self):
        X, R = _make_history(100, N=5)
        est = KNNMomentEstimator(k=10)
        est.fit_from_history(X, R)
        result = est.estimate_moments(X[0])
        assert result is not None
        _, Sigma = result
        eigvals = np.linalg.eigvalsh(Sigma)
        assert (eigvals >= -1e-8).all()

    def test_sigma_is_symmetric(self):
        X, R = _make_history(100, N=5)
        est = KNNMomentEstimator(k=10)
        est.fit_from_history(X, R)
        result = est.estimate_moments(X[0])
        assert result is not None
        _, Sigma = result
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)

    def test_returns_none_when_insufficient_history(self):
        X, R = _make_history(5, N=5)
        est = KNNMomentEstimator(k=10)
        est.fit_from_history(X, R)
        result = est.estimate_moments(X[0])
        assert result is None  # 5 < 10 + 5 = 15

    def test_returns_none_with_empty_history(self):
        est = KNNMomentEstimator(k=5)
        result = est.estimate_moments(np.zeros(15))
        assert result is None

    def test_update_accumulates_history(self):
        est = KNNMomentEstimator(k=5)
        for i in range(10):
            est.update(np.zeros(15), np.zeros(5))
        assert est.X_history.shape[0] == 10

    def test_consistent_results_for_same_query(self):
        X, R = _make_history(100, N=5)
        est = KNNMomentEstimator(k=10)
        est.fit_from_history(X, R)
        r1 = est.estimate_moments(X[0])
        r2 = est.estimate_moments(X[0])
        assert r1 is not None and r2 is not None
        np.testing.assert_allclose(r1[0], r2[0], atol=1e-12)

    def test_different_queries_differ(self):
        rng = np.random.default_rng(42)
        X, R = _make_history(200, N=5)
        est = KNNMomentEstimator(k=5)
        est.fit_from_history(X, R)
        r1 = est.estimate_moments(X[10])
        r2 = est.estimate_moments(X[150])
        assert r1 is not None and r2 is not None
        # Means should differ for very different queries
        assert not np.allclose(r1[0], r2[0])
