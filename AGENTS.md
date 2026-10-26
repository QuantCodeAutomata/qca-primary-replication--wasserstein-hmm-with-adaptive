# Repository Knowledge — Quantitative Finance Project

## Project Layout

```
src/            Core library modules (optimizer, knn_baseline, metrics)
data/           Raw and processed data files
exp/            Experiment scripts and notebooks
results/        Output artefacts (weights, metrics, plots)
config/         YAML/JSON configuration files
tests/          Unit and integration tests
```

## Core Modules (`src/`)

### `optimizer.py` — Transaction-Cost-Aware MVO
- **Constants**: `GAMMA=1.0`, `TAU=0.01`, `W_MAX=0.6`, `N_ASSETS=5`
- **`solve_mvo(mu, Sigma, w_prev, gamma, tau, w_max)`**
  - Formulation: maximize `mu'w - gamma·w'Σw - tau·‖w − w_prev‖₁`
  - Constraints: `1'w = 1`, `w ≥ 0`, `w ≤ w_max`
  - Uses `cp.psd_wrap(Sigma)` for the quad-form to signal PSD structure to cvxpy.
  - Falls back to `w_prev` on solver failure (logs warning).
  - Always clips negatives and re-normalises before returning.
  - Logs a warning if `w_max < 1/N` (potentially infeasible).
- **`equal_weights(n)`** — returns `np.ones(n)/n`.

### `knn_baseline.py` — KNN Conditional Moment Estimator
- **Constant**: `K_NEIGHBORS=30`
- **`KNNMomentEstimator(k, metric)`**
  - Stores `X_history` (lagged features) and `R_history` (returns) as 2D numpy arrays.
  - `update(X_new, R_new)` — appends one observation; handles 1D or 2D inputs via `np.atleast_2d`.
  - `fit_from_history(X_hist, R_hist)` — bulk initialisation.
  - `estimate_moments(x_query, n_assets)` — returns `(mu_cond, Sigma_cond)` or `None` if `T < k+5`.
    - Re-fits `StandardScaler` on `X_history` each call (rolling window semantics).
    - Ledoit-Wolf shrinkage covariance on neighbour returns when `k ≥ 2`.
    - Falls back to `np.eye(n_assets) * 1e-4` when `k < 2`.

### `metrics.py` — Portfolio Performance Metrics
- All scalar functions accept `np.ndarray | pd.Series`.
- `compute_sharpe` / `compute_sortino` / `compute_annualized_return` / `compute_annualized_vol` — standard annualised stats (default 252 trading days).
- `compute_max_drawdown` — uses `(1+r).cumprod()` then running-max ratio; returns a **non-positive** float.
- `compute_sortino` — downside std uses only `r < 0` observations.
- `compute_turnover_stats(turnovers)` — dict: `mean, median, std, p95, frac_above_1pct, frac_above_5pct`.
- `compute_effective_n(weights_matrix)` — inverse-HHI per row; accepts `(T, N)` array or DataFrame.
- `compute_full_metrics(returns, weights, label)` — aggregates everything; turnover = `0.5 · Σ|Δw|`.

## Data Layer (`data/`)

### `loader.py` — Yahoo Finance Price Loader
- **`TICKER_MANIFEST`**: `{SPX: SPY, BOND: TLT, GOLD: GLD, OIL: USO, USD: UUP}`
- **`download_prices(start, end, tickers)`**
  - Calls `yfinance.download(..., auto_adjust=True, progress=False)`.
  - yfinance ≥1.4 returns a MultiIndex DataFrame; extracts `raw["Close"]`.
  - Renames ticker symbols → asset names; enforces manifest key order.
  - Logs row counts; warns on per-asset missing-value counts; drops incomplete rows.
  - Raises `ValueError` if the download is empty.
- **`load_prices(cache_path, force_reload)`**
  - Cache-first: reads from parquet on cache hit, otherwise calls `download_prices` and saves.
  - Default cache: `data/prices.parquet`; creates parent directory with `os.makedirs(..., exist_ok=True)`.
  - Parquet support requires `pyarrow` (install `pyarrow<15` on NumPy 1.26 environments).

## Feature Engineering (`src/`)

### `features.py` — Strictly Causal Feature Engineering
- **Constants**: `ROLL_VOL_WINDOW=60`, `ROLL_MEAN_WINDOW=20`, `N_ASSETS=5`
- **`compute_log_returns(prices)`** — `log(P_t/P_{t-1})`, drops first NaN row.
- **`compute_rolling_features(returns, vol_window, mean_window)`** — returns `(rolling_vol, rolling_mean)` **unshifted**; caller must lag for causality.
- **`build_feature_matrix(returns, rolling_vol, rolling_mean)`**
  - Concatenates and interleaves columns as `[{asset}_ret, {asset}_vol, {asset}_mom]` per asset.
  - Drops warm-up NaN rows; output shape `(T − warm_up, 3·N_assets)`.
- **`build_causal_features(prices, vol_window, mean_window)`** — main entry point.
  - Returns `(returns, features, lagged_features)` all sharing the same date index.
  - `lagged_features = features.shift(1)` then inner-joined + `dropna()` to strip leading NaN.
  - Causality invariant: `lagged_features.iloc[t] == features.iloc[t-1]` for all valid `t`.
  - With 300 days of prices and defaults: 299 return rows → 240 feature rows → **239 aligned rows**.

### `hmm_model.py` — Gaussian HMM Fitting and Adaptive Order Selection
- **Constants**: `K_MIN=2`, `K_MAX=8`, `COMPLEXITY_PENALTY=0.5`, `VAL_SLICE_LENGTH=60`, `ORDER_SELECT_FREQ=5`
- **`fit_gaussian_hmm(X, n_components, random_state, n_iter, tol)`**
  - Fits `GaussianHMM(covariance_type='full', min_covar=1e-4)` from hmmlearn.
  - Catches all exceptions; logs warning and returns `None` on failure.
  - Suppresses hmmlearn convergence warnings via `warnings.catch_warnings`.
- **`compute_predictive_loglik(model, X_val)`**
  - One-step-ahead predictive log-likelihood loop over validation slice.
  - Initialises with `model.startprob_`; propagates via `filtered @ transmat`.
  - Emission densities from `scipy.stats.multivariate_normal.logpdf`.
  - Uses `scipy.special.logsumexp` throughout; Bayes update via log-normalisation.
- **`select_hmm_order(X_train, X_val, k_min, k_max, penalty, random_state)`**
  - Penalised score = `PredLL(K) − penalty × K`; returns `(best_K, scores_dict)`.
  - Falls back to `k_min` when all fits fail.
- **`AdaptiveHMMSelector`** — skips re-selection unless `t % select_freq == 0`.
  - Minimum history guard: `len(X_history) > val_slice_length + k_min + 1`.
  - Splits history as `X_train = X_history[:-val_slice_length]`, `X_val = X_history[-val_slice_length:]`.
- **`get_predictive_state_probs(model, X_history)`**
  - Uses `model.predict_proba(X_history)[-1] @ model.transmat_`; renormalises.
- **`extract_hmm_components(model)`**
  - Returns `(means, covars)` with covars symmetrised via `(C + C.T) / 2`.

### `wasserstein.py` — Wasserstein Template-Based Regime Tracking
- **`psd_sqrt(M)`** — eigendecomposition, clips eigenvalues at `1e-10`, symmetrises result.
- **`psd_project(M)`** — clips negative eigenvalues to 0, symmetrises result.
- **`gaussian_w2_squared(mu1, Sigma1, mu2, Sigma2)`**
  - Formula: `‖μ₁−μ₂‖² + Tr(Σ₁ + Σ₂ − 2·(Σ₂^½ Σ₁ Σ₂^½)^½)`.
  - Inner matrix symmetrised before square root; trace term clipped at 0.
- **`WassersteinTemplateTracker`**
  - `n_templates` (G), `eta`, `feature_dim` (D), `n_assets` (N).
  - `template_means: (G, D)`, `template_covars: (G, D, D)`.
  - `initialize_from_hmm`: cycles components round-robin when K < G; takes first G when K > G.
  - `assign_components_to_templates`: argmin W2 distance from each component to each template.
  - `aggregate_template_probs`: sums state_probs into template buckets then normalises.
  - `update_templates`: EMA with normalised within-group weights; applies `psd_project` to updated covariance.
  - `get_return_moments`: mixture over first-N-feature block; returns PSD-projected Sigma.
  - `step`: assign → aggregate → update → record `self.dominant` (argmax) → return moments.

## Dependencies
- `cvxpy >= 1.7`, `numpy`, `pandas`, `scikit-learn`, `yfinance`, `pyarrow<15`
- `hmmlearn`, `scipy` (stats, linalg, special)
- Python ≥ 3.10 (uses `X | Y` union type-hint syntax throughout)

## Conventions
- All modules use `logging.getLogger(__name__)` — no `print` statements in library code.
- Type hints follow NumPy-style docstrings.
- `src/` must be on `sys.path` (or installed as a package) for imports to work.
