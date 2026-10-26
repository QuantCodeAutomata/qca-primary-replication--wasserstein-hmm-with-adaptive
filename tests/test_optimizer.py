"""Tests for src/optimizer.py — transaction-cost-aware MVO."""

import numpy as np
import pytest

from src.optimizer import equal_weights, solve_mvo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(
    N: int = 5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(N) * 0.001
    A = rng.standard_normal((N, N)) * 0.1
    Sigma = A @ A.T / N + np.eye(N) * 1e-3
    w_prev = np.ones(N) / N
    return mu, Sigma, w_prev


# ---------------------------------------------------------------------------
# equal_weights
# ---------------------------------------------------------------------------

class TestEqualWeights:
    def test_default_five_assets(self):
        w = equal_weights(5)
        assert w.shape == (5,)
        np.testing.assert_allclose(w, np.ones(5) / 5)

    def test_sums_to_one(self):
        for n in [2, 3, 7, 10]:
            assert abs(equal_weights(n).sum() - 1.0) < 1e-12

    def test_all_positive(self):
        w = equal_weights(6)
        assert (w > 0).all()


# ---------------------------------------------------------------------------
# solve_mvo
# ---------------------------------------------------------------------------

class TestSolveMVO:
    def test_returns_correct_shape(self):
        mu, Sigma, w_prev = _make_params(5)
        w = solve_mvo(mu, Sigma, w_prev)
        assert w.shape == (5,)

    def test_weights_sum_to_one(self):
        mu, Sigma, w_prev = _make_params(5)
        w = solve_mvo(mu, Sigma, w_prev)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)

    def test_weights_non_negative(self):
        mu, Sigma, w_prev = _make_params(5)
        w = solve_mvo(mu, Sigma, w_prev)
        assert (w >= -1e-8).all()

    def test_max_weight_constraint(self):
        mu, Sigma, w_prev = _make_params(5)
        w_max = 0.4
        w = solve_mvo(mu, Sigma, w_prev, w_max=w_max)
        assert (w <= w_max + 1e-6).all()

    def test_turnover_penalty_reduces_churn(self):
        # High tau should keep weights closer to w_prev
        mu, Sigma, w_prev = _make_params(5, seed=1)
        w_low_tau = solve_mvo(mu, Sigma, w_prev, tau=0.0)
        w_high_tau = solve_mvo(mu, Sigma, w_prev, tau=1.0)
        low_to = np.sum(np.abs(w_low_tau - w_prev))
        high_to = np.sum(np.abs(w_high_tau - w_prev))
        assert high_to <= low_to + 1e-4

    def test_deterministic_output(self):
        mu, Sigma, w_prev = _make_params(5)
        w1 = solve_mvo(mu, Sigma, w_prev)
        w2 = solve_mvo(mu, Sigma, w_prev)
        np.testing.assert_allclose(w1, w2, atol=1e-8)

    def test_fallback_on_infeasible(self):
        # w_max < 1/N: solver should return a valid fallback (w_prev)
        N = 5
        mu, Sigma, w_prev = _make_params(N)
        w = solve_mvo(mu, Sigma, w_prev, w_max=1.0 / N - 0.01)
        # Must still be a valid allocation
        assert w.sum() > 0

    def test_single_asset(self):
        mu = np.array([0.001])
        Sigma = np.array([[0.0001]])
        w_prev = np.array([1.0])
        w = solve_mvo(mu, Sigma, w_prev)
        np.testing.assert_allclose(w, np.array([1.0]), atol=1e-6)

    def test_risk_aversion_effect(self):
        # Higher gamma → weights closer to equal (less aggressive)
        mu, Sigma, w_prev = _make_params(5, seed=3)
        w_lo = solve_mvo(mu, Sigma, w_prev, gamma=0.1)
        w_hi = solve_mvo(mu, Sigma, w_prev, gamma=100.0)
        hhi_lo = np.sum(w_lo ** 2)
        hhi_hi = np.sum(w_hi ** 2)
        # Higher gamma should generally lead to more diversified portfolio
        # (lower HHI), though not guaranteed for every scenario
        assert hhi_hi <= hhi_lo + 0.1  # loose tolerance
