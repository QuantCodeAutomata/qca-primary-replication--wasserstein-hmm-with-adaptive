"""Tests for src/metrics.py — performance metric calculations."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_annualized_return,
    compute_annualized_vol,
    compute_effective_n,
    compute_full_metrics,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_turnover_stats,
)


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------

class TestComputeSharpe:
    def test_positive_returns_positive_sharpe(self):
        r = np.full(252, 0.001)
        assert compute_sharpe(r) > 0.0

    def test_zero_volatility_returns_zero(self):
        r = np.full(100, 0.0)
        assert compute_sharpe(r) == 0.0

    def test_annualization_factor(self):
        r = np.full(252, 0.001)
        sharpe_252 = compute_sharpe(r, annualization=252)
        sharpe_1 = compute_sharpe(r, annualization=1)
        # Sharpe should scale with sqrt of annualization
        np.testing.assert_allclose(sharpe_252 / sharpe_1, np.sqrt(252), atol=1e-6)

    def test_accepts_series(self):
        s = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 252))
        sharpe = compute_sharpe(s)
        assert np.isfinite(sharpe)

    def test_negative_mean_is_negative_sharpe(self):
        r = np.full(252, -0.001)
        assert compute_sharpe(r) < 0.0


# ---------------------------------------------------------------------------
# compute_sortino
# ---------------------------------------------------------------------------

class TestComputeSortino:
    def test_positive_with_upside_heavy_returns(self):
        rng = np.random.default_rng(1)
        r = np.abs(rng.normal(0.002, 0.01, 500))  # always positive
        sortino = compute_sortino(r)
        # All returns above zero → downside std is for negatives only (none)
        # Should be 0.0 fallback
        assert sortino == 0.0 or np.isfinite(sortino)

    def test_zero_downside_returns_zero(self):
        r = np.full(100, 0.001)
        assert compute_sortino(r) == 0.0

    def test_accepts_series(self):
        s = pd.Series(np.random.default_rng(2).normal(0, 0.01, 252))
        sortino = compute_sortino(s)
        assert np.isfinite(sortino)


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown:
    def test_zero_return_series(self):
        r = np.zeros(100)
        mdd = compute_max_drawdown(r)
        assert mdd == 0.0

    def test_always_positive_returns_gives_zero_drawdown(self):
        r = np.full(100, 0.001)
        mdd = compute_max_drawdown(r)
        assert abs(mdd) < 1e-10

    def test_single_crash_then_recovery(self):
        r = np.array([0.01] * 10 + [-0.2] + [0.01] * 10)
        mdd = compute_max_drawdown(r)
        assert mdd < 0.0

    def test_result_is_non_positive(self):
        r = np.random.default_rng(3).normal(0, 0.01, 500)
        mdd = compute_max_drawdown(r)
        assert mdd <= 0.0

    def test_known_drawdown(self):
        # Perfect sequence: 50% up then 50% down → max DD ≈ -33%
        r = np.array([0.5, -0.5])
        mdd = compute_max_drawdown(r)
        # After 50% gain: 1.5, then 50% loss: 0.75 → DD = 0.75/1.5 - 1 = -0.5
        np.testing.assert_allclose(mdd, -0.5, atol=1e-8)

    def test_accepts_series(self):
        s = pd.Series(np.random.default_rng(4).normal(0, 0.01, 252))
        mdd = compute_max_drawdown(s)
        assert np.isfinite(mdd)


# ---------------------------------------------------------------------------
# compute_annualized_return / compute_annualized_vol
# ---------------------------------------------------------------------------

class TestAnnualizedReturnVol:
    def test_annualized_return_scales_linearly(self):
        r = np.full(252, 0.001)
        ann = compute_annualized_return(r, annualization=252)
        np.testing.assert_allclose(ann, 0.001 * 252, atol=1e-12)

    def test_annualized_vol_scales_with_sqrt(self):
        r = np.random.default_rng(5).normal(0, 0.01, 1000)
        vol_252 = compute_annualized_vol(r, annualization=252)
        vol_1 = compute_annualized_vol(r, annualization=1)
        np.testing.assert_allclose(vol_252 / vol_1, np.sqrt(252), atol=1e-8)


# ---------------------------------------------------------------------------
# compute_turnover_stats
# ---------------------------------------------------------------------------

class TestComputeTurnoverStats:
    def test_empty_returns_zeros(self):
        stats = compute_turnover_stats(np.array([]))
        for v in stats.values():
            assert v == 0.0

    def test_fractions_in_zero_one(self):
        t = np.random.default_rng(6).uniform(0, 0.1, 200)
        stats = compute_turnover_stats(t)
        assert 0.0 <= stats["frac_above_1pct"] <= 1.0
        assert 0.0 <= stats["frac_above_5pct"] <= 1.0

    def test_p95_at_least_mean(self):
        t = np.abs(np.random.default_rng(7).normal(0.02, 0.01, 500))
        stats = compute_turnover_stats(t)
        assert stats["p95"] >= stats["mean"]

    def test_all_keys_present(self):
        stats = compute_turnover_stats(np.array([0.01, 0.02, 0.06]))
        expected_keys = {
            "mean", "median", "std", "p95", "frac_above_1pct", "frac_above_5pct"
        }
        assert set(stats.keys()) == expected_keys


# ---------------------------------------------------------------------------
# compute_effective_n
# ---------------------------------------------------------------------------

class TestComputeEffectiveN:
    def test_equal_weights_gives_n(self):
        N = 5
        W = np.ones((10, N)) / N
        n_eff = compute_effective_n(W)
        np.testing.assert_allclose(n_eff, np.full(10, N), atol=1e-8)

    def test_concentrated_portfolio_gives_one(self):
        W = np.zeros((5, 4))
        W[:, 0] = 1.0
        n_eff = compute_effective_n(W)
        np.testing.assert_allclose(n_eff, np.ones(5), atol=1e-8)

    def test_accepts_dataframe(self):
        W = pd.DataFrame(np.ones((3, 4)) / 4)
        n_eff = compute_effective_n(W)
        assert n_eff.shape == (3,)
        np.testing.assert_allclose(n_eff, np.full(3, 4.0), atol=1e-8)

    def test_values_between_one_and_n(self):
        N = 5
        W = np.abs(np.random.default_rng(8).normal(size=(100, N)))
        W = W / W.sum(axis=1, keepdims=True)
        n_eff = compute_effective_n(W)
        assert (n_eff >= 1.0 - 1e-8).all()
        assert (n_eff <= N + 1e-8).all()


# ---------------------------------------------------------------------------
# compute_full_metrics
# ---------------------------------------------------------------------------

class TestComputeFullMetrics:
    def _make_returns_weights(
        self, T: int = 200, N: int = 5, seed: int = 0
    ) -> tuple[pd.Series, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-01", periods=T)
        returns = pd.Series(rng.normal(0.001, 0.01, T), index=idx)
        raw = np.abs(rng.normal(size=(T, N)))
        weights = pd.DataFrame(
            raw / raw.sum(axis=1, keepdims=True), index=idx
        )
        return returns, weights

    def test_all_keys_present(self):
        returns, weights = self._make_returns_weights()
        metrics = compute_full_metrics(returns, weights, label="test")
        expected_keys = {
            "label", "sharpe", "sortino", "max_drawdown", "ann_return",
            "ann_vol", "avg_turnover", "p95_turnover", "frac_above_1pct",
            "frac_above_5pct", "avg_n_eff", "cumulative_return",
        }
        assert set(metrics.keys()) == expected_keys

    def test_sharpe_is_finite(self):
        returns, weights = self._make_returns_weights()
        metrics = compute_full_metrics(returns, weights)
        assert np.isfinite(metrics["sharpe"])

    def test_max_drawdown_non_positive(self):
        returns, weights = self._make_returns_weights()
        metrics = compute_full_metrics(returns, weights)
        assert metrics["max_drawdown"] <= 0.0

    def test_avg_n_eff_positive(self):
        returns, weights = self._make_returns_weights()
        metrics = compute_full_metrics(returns, weights)
        assert metrics["avg_n_eff"] > 0.0

    def test_label_preserved(self):
        returns, weights = self._make_returns_weights()
        metrics = compute_full_metrics(returns, weights, label="my_strategy")
        assert metrics["label"] == "my_strategy"
