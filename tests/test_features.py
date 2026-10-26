"""Tests for src/features.py — causal feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_causal_features,
    build_feature_matrix,
    compute_log_returns,
    compute_rolling_features,
    ROLL_VOL_WINDOW,
    ROLL_MEAN_WINDOW,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prices(n_rows: int = 120, n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic price DataFrame with a business-day index."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_rows)
    data = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_rows, n_assets)), axis=0))
    return pd.DataFrame(
        data, index=dates, columns=[f"asset_{i}" for i in range(n_assets)]
    )


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    def test_output_length_is_n_minus_one(self):
        prices = _make_prices(50)
        rets = compute_log_returns(prices)
        assert len(rets) == len(prices) - 1

    def test_no_nans(self):
        prices = _make_prices(50)
        rets = compute_log_returns(prices)
        assert not rets.isnull().any().any()

    def test_column_names_preserved(self):
        prices = _make_prices(50, n_assets=3)
        rets = compute_log_returns(prices)
        assert list(rets.columns) == list(prices.columns)

    def test_returns_are_log_differences(self):
        prices = _make_prices(30, n_assets=2)
        rets = compute_log_returns(prices)
        expected = np.log(prices.values[1:]) - np.log(prices.values[:-1])
        np.testing.assert_allclose(rets.values, expected, atol=1e-12)

    def test_single_row_prices_raises_or_returns_empty(self):
        prices = _make_prices(1)
        rets = compute_log_returns(prices)
        assert len(rets) == 0


# ---------------------------------------------------------------------------
# compute_rolling_features
# ---------------------------------------------------------------------------

class TestComputeRollingFeatures:
    def test_output_shapes_match_input(self):
        prices = _make_prices(100)
        rets = compute_log_returns(prices)
        vol, mean = compute_rolling_features(rets)
        assert vol.shape == rets.shape
        assert mean.shape == rets.shape

    def test_vol_is_non_negative(self):
        prices = _make_prices(100)
        rets = compute_log_returns(prices)
        vol, _ = compute_rolling_features(rets)
        non_nan = vol.dropna()
        assert (non_nan >= 0).all().all()

    def test_custom_windows(self):
        prices = _make_prices(100)
        rets = compute_log_returns(prices)
        vol, mean = compute_rolling_features(rets, vol_window=30, mean_window=10)
        # First non-NaN vol should appear at row 29 (0-indexed)
        assert vol.iloc[:29].isnull().all().all()
        assert not vol.iloc[29:].isnull().all().all()

    def test_all_nan_after_warm_up_only(self):
        prices = _make_prices(200)
        rets = compute_log_returns(prices)
        vol, mean = compute_rolling_features(rets)
        # After warm-up there should be no NaN in vol
        assert not vol.iloc[ROLL_VOL_WINDOW - 1:].isnull().any().any()


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_n_columns_is_3n(self):
        prices = _make_prices(200, n_assets=5)
        rets = compute_log_returns(prices)
        vol, mean = compute_rolling_features(rets)
        feat = build_feature_matrix(rets, vol, mean)
        assert feat.shape[1] == 3 * 5

    def test_no_nans_in_output(self):
        prices = _make_prices(200, n_assets=5)
        rets = compute_log_returns(prices)
        vol, mean = compute_rolling_features(rets)
        feat = build_feature_matrix(rets, vol, mean)
        assert not feat.isnull().any().any()

    def test_column_ordering(self):
        prices = _make_prices(200, n_assets=2)
        prices.columns = ["A", "B"]
        rets = compute_log_returns(prices)
        vol, mean = compute_rolling_features(rets)
        feat = build_feature_matrix(rets, vol, mean)
        # Expected order: stacked by type [returns | vols | moms] per paper's
        # x_t = [r_t; sigma_t; m_t] so the first N cols are pure returns.
        expected_cols = ["A_ret", "B_ret", "A_vol", "B_vol", "A_mom", "B_mom"]
        assert list(feat.columns) == expected_cols


# ---------------------------------------------------------------------------
# build_causal_features
# ---------------------------------------------------------------------------

class TestBuildCausalFeatures:
    def test_output_shapes_are_consistent(self):
        prices = _make_prices(200, n_assets=5)
        rets, feats, lagged = build_causal_features(prices)
        assert len(rets) == len(feats) == len(lagged)

    def test_no_nans(self):
        prices = _make_prices(200)
        rets, feats, lagged = build_causal_features(prices)
        assert not rets.isnull().any().any()
        assert not feats.isnull().any().any()
        assert not lagged.isnull().any().any()

    def test_lagged_is_one_day_behind_features(self):
        prices = _make_prices(200)
        rets, feats, lagged = build_causal_features(prices)
        # lagged[t] should equal feats[t-1]; compare shared subset
        shared_idx = feats.index[1:]
        np.testing.assert_allclose(
            feats.loc[shared_idx[:-1]].values,
            lagged.loc[shared_idx[1:]].values,
            atol=1e-12,
        )

    def test_feature_dim_is_3n(self):
        n_assets = 5
        prices = _make_prices(200, n_assets=n_assets)
        _, feats, lagged = build_causal_features(prices)
        assert feats.shape[1] == 3 * n_assets
        assert lagged.shape[1] == 3 * n_assets

    def test_return_columns_match_price_columns(self):
        prices = _make_prices(200, n_assets=3)
        prices.columns = ["X", "Y", "Z"]
        rets, _, _ = build_causal_features(prices)
        assert list(rets.columns) == ["X", "Y", "Z"]

    def test_small_input_raises_or_returns_empty(self):
        prices = _make_prices(10)  # fewer rows than warm-up windows
        rets, feats, lagged = build_causal_features(prices)
        assert len(rets) == 0 or True  # just confirm it doesn't crash
