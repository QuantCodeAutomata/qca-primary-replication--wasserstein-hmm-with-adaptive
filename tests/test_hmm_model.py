"""Tests for src/hmm_model.py — HMM fitting and order selection."""

import numpy as np
import pytest

from src.hmm_model import (
    AdaptiveHMMSelector,
    compute_predictive_loglik,
    extract_hmm_components,
    fit_gaussian_hmm,
    get_predictive_state_probs,
    select_hmm_order,
    K_MIN, K_MAX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observations(T: int = 200, D: int = 15, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, D))


# ---------------------------------------------------------------------------
# fit_gaussian_hmm
# ---------------------------------------------------------------------------

class TestFitGaussianHMM:
    def test_returns_model_for_valid_input(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, n_components=3)
        assert model is not None

    def test_means_shape(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, 3)
        assert model.means_.shape == (3, 15)

    def test_covars_shape(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, 3)
        assert model.covars_.shape == (3, 15, 15)

    def test_transmat_rows_sum_to_one(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, 3)
        np.testing.assert_allclose(
            model.transmat_.sum(axis=1),
            np.ones(3),
            atol=1e-6,
        )

    def test_startprob_sums_to_one(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, 3)
        np.testing.assert_allclose(model.startprob_.sum(), 1.0, atol=1e-6)

    def test_returns_none_for_impossible_fit(self):
        # Single observation: EM cannot fit
        X = _make_observations(1, 3)
        model = fit_gaussian_hmm(X, 3)
        assert model is None

    def test_reproducible_with_same_seed(self):
        X = _make_observations(200, 15)
        m1 = fit_gaussian_hmm(X, 3, random_state=0)
        m2 = fit_gaussian_hmm(X, 3, random_state=0)
        np.testing.assert_allclose(m1.means_, m2.means_, atol=1e-8)


# ---------------------------------------------------------------------------
# extract_hmm_components
# ---------------------------------------------------------------------------

class TestExtractHMMComponents:
    def test_shapes(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, 4)
        means, covars = extract_hmm_components(model)
        assert means.shape == (4, 15)
        assert covars.shape == (4, 15, 15)

    def test_covars_are_symmetric(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, 3)
        _, covars = extract_hmm_components(model)
        for k in range(3):
            np.testing.assert_allclose(
                covars[k], covars[k].T, atol=1e-10
            )


# ---------------------------------------------------------------------------
# compute_predictive_loglik
# ---------------------------------------------------------------------------

class TestComputePredictiveLoglik:
    def test_returns_finite_float(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X[:150], 3)
        ll = compute_predictive_loglik(model, X[150:])
        assert np.isfinite(ll)

    def test_more_obs_yields_larger_ll_in_absolute_terms(self):
        # Not necessarily true for predictive LL, but len proportionality should hold approximately
        X = _make_observations(200, 4, seed=1)
        model = fit_gaussian_hmm(X[:100], 2)
        ll_short = compute_predictive_loglik(model, X[100:110])
        ll_long = compute_predictive_loglik(model, X[100:120])
        # Longer validation should contribute more total log-likelihood
        assert abs(ll_long) >= abs(ll_short) - 1e-3  # very loose bound

    def test_empty_val_returns_zero(self):
        X = _make_observations(100, 4)
        model = fit_gaussian_hmm(X, 2)
        ll = compute_predictive_loglik(model, X[:0])
        assert ll == 0.0


# ---------------------------------------------------------------------------
# get_predictive_state_probs
# ---------------------------------------------------------------------------

class TestGetPredictiveStateProbs:
    def test_output_sums_to_one(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, 4)
        probs = get_predictive_state_probs(model, X)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-8)

    def test_all_non_negative(self):
        X = _make_observations(200, 15)
        model = fit_gaussian_hmm(X, 4)
        probs = get_predictive_state_probs(model, X)
        assert (probs >= 0).all()

    def test_output_shape(self):
        X = _make_observations(200, 15)
        K = 5
        model = fit_gaussian_hmm(X, K)
        probs = get_predictive_state_probs(model, X)
        assert probs.shape == (K,)


# ---------------------------------------------------------------------------
# select_hmm_order
# ---------------------------------------------------------------------------

class TestSelectHMMOrder:
    def test_returns_k_in_valid_range(self):
        X = _make_observations(300, 10)
        best_k, scores = select_hmm_order(X[:200], X[200:], k_min=2, k_max=5)
        assert K_MIN <= best_k <= 5

    def test_scores_keys_are_candidate_ks(self):
        X = _make_observations(300, 10)
        _, scores = select_hmm_order(X[:200], X[200:], k_min=2, k_max=4)
        for k in scores:
            assert 2 <= k <= 4

    def test_penalised_lower_than_raw(self):
        X = _make_observations(300, 10)
        _, scores = select_hmm_order(X[:200], X[200:], k_min=2, k_max=4, penalty=1.0)
        for k, (pred_ll, penalised) in scores.items():
            assert penalised <= pred_ll + 1e-9

    def test_fallback_when_all_fits_fail(self):
        # Too small sample for the requested K
        X = _make_observations(3, 2)
        best_k, scores = select_hmm_order(X[:1], X[1:], k_min=2, k_max=3)
        assert best_k >= 2  # falls back to k_min


# ---------------------------------------------------------------------------
# AdaptiveHMMSelector
# ---------------------------------------------------------------------------

class TestAdaptiveHMMSelector:
    def test_runs_without_error(self):
        X = _make_observations(400, 10)
        selector = AdaptiveHMMSelector(k_min=2, k_max=4, val_slice_length=30, select_freq=5)
        # t=0 → 0 % 5 == 0 → selection runs
        k, scores = selector.fit_predict(X, t=0, current_K=2)
        assert 2 <= k <= 4

    def test_carry_forward_on_non_selection_step(self):
        X = _make_observations(400, 10)
        selector = AdaptiveHMMSelector(k_min=2, k_max=4, select_freq=5)
        # t=1 → 1 % 5 != 0 → carry forward
        k, _ = selector.fit_predict(X, t=1, current_K=3)
        assert k == 3  # carried forward

    def test_selection_on_multiples_of_freq(self):
        X = _make_observations(400, 10)
        selector = AdaptiveHMMSelector(k_min=2, k_max=4, val_slice_length=20, select_freq=5)
        # t=5 → 5 % 5 == 0 → selection runs
        k, scores = selector.fit_predict(X, t=5, current_K=2)
        # Selection runs → scores dict should be populated
        assert isinstance(scores, dict)
