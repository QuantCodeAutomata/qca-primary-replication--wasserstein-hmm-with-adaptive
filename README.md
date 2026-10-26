# Wasserstein HMM Adaptive Regime Allocator — Primary Replication

Replication repository for a cross-asset portfolio allocation system that combines:

- **Gaussian Hidden Markov Model** with adaptive state-count selection (penalised predictive log-likelihood)
- **2-Wasserstein template tracking** for persistent regime identity across daily HMM re-fits
- **Transaction-cost-aware Mean-Variance Optimisation** (CVXPY, L1 turnover penalty)
- **KNN conditional-moment estimator** as a non-parametric baseline using the same MVO layer

Evaluated over a strictly out-of-sample period (≈ 690 trading days, May 2023 → Jan 2026) on a 5-asset cross-asset universe: SPY, TLT, GLD, USO, UUP.

---

## Repository Structure

```
├── config/
│   └── manifest.yaml          # All hyperparameters (G, eta, gamma, tau, K range, etc.)
├── data/
│   └── loader.py              # Yahoo Finance price download + Parquet caching
├── src/
│   ├── features.py            # Log returns, rolling vol/momentum, causal feature lag
│   ├── hmm_model.py           # GaussianHMM fitting, adaptive K selection (pred LL)
│   ├── wasserstein.py         # 2-Wasserstein distance, template matching + updating
│   ├── optimizer.py           # Transaction-cost-aware MVO (CVXPY)
│   ├── knn_baseline.py        # KNN conditional moment estimator (Ledoit-Wolf cov)
│   ├── metrics.py             # Sharpe, Sortino, MaxDD, turnover, N_eff
│   └── strategy.py            # Full HMM+template+MVO strategy runner
├── exp/
│   ├── exp_1_primary.py       # EXP1: HMM+MVO vs KNN+MVO head-to-head
│   ├── exp_2_order_selection.py  # EXP2: Isolated adaptive K selection evaluation
│   ├── exp_3_template_tracking.py  # EXP3: Template stability vs raw HMM labels
│   └── exp_4_benchmarks.py    # EXP4: HMM vs SPX B&H vs equal-weight
├── tests/
│   ├── test_features.py
│   ├── test_hmm_model.py
│   ├── test_wasserstein.py
│   ├── test_optimizer.py
│   ├── test_metrics.py
│   └── test_knn_baseline.py
├── results/
│   ├── RESULTS.md             # Full experiment results with metrics and discussion
│   ├── exp1_*.png/csv         # EXP1 figures and data
│   ├── exp2_*.png/csv         # EXP2 figures and data
│   ├── exp3_*.png/csv         # EXP3 figures and data
│   └── exp4_*.png/csv         # EXP4 figures and data
├── logs/                      # Experiment run logs
└── run_all.sh                 # Sequential experiment runner
```

---

## Asset Universe (Ticker Manifest)

| Proxy | Ticker | Description |
|-------|--------|-------------|
| SPX | SPY | SPDR S&P 500 ETF |
| BOND | TLT | iShares 20+ Year Treasury |
| GOLD | GLD | SPDR Gold Shares |
| OIL | USO | United States Oil Fund |
| USD | UUP | Invesco DB USD Index Bullish |

---

## Setup and Running

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn hmmlearn cvxpy yfinance matplotlib seaborn
```

### Download data

```bash
python data/loader.py
```

Prices are cached to `data/prices.parquet` for offline use.

### Run all experiments

```bash
bash run_all.sh
```

Or individually:

```bash
python exp/exp_1_primary.py
python exp/exp_2_order_selection.py
python exp/exp_3_template_tracking.py
python exp/exp_4_benchmarks.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## Key Hyperparameters

See `config/manifest.yaml` for the full manifest. Core settings:

| Parameter | Value |
|-----------|-------|
| Templates G | 6 |
| Template smoothing η | 0.05 |
| MVO risk aversion γ | 1.0 |
| Turnover penalty τ | 0.01 |
| Weight cap w_max | 0.60 |
| K range | 2–8 |
| Order-selection freq F_K | 5 days |
| Validation slice V_len | 60 days |
| KNN neighbours | 20 |
| Training split t_0 | 2023-05-01 |

---

## Results Summary

See [`results/RESULTS.md`](results/RESULTS.md) for full metrics and figures.

| Hypothesis | Result |
|-----------|--------|
| HMM Sharpe > KNN Sharpe | ✅ 0.652 vs 0.555 |
| HMM MaxDD better than KNN | ✅ −17.2% vs −22.8% |
| HMM turnover << KNN | ✅ 6.2× lower (0.089 vs 0.552) |
| Adaptive K_t non-constant | ✅ Mean 4.41, 105 transitions |
| Adaptive K beats all fixed-K | ✅ +4 930 pred LL vs best fixed |
| Templates reduce label switching | ✅ 49.6% fewer switches |
| Dormant template(s) | ✅ Templates 1 & 2 (<0.5% mass) |
| HMM > EW > SPX Sharpe ranking | ❌ Ordering reversed in 2023–2026 bull run |

---

## Implementation Notes

- **Strict causality**: features at time t use only data through t−1
- **Expanding window**: every HMM re-fit uses the full available history
- **PSD enforcement**: eigenvalue clipping at 1e-6 for all covariance matrices
- **Wasserstein distance**: closed-form Gaussian formula via `scipy.linalg.sqrtm`
- **MVO solver**: CVXPY + CLARABEL with L1 turnover penalty via auxiliary variables
