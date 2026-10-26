"""
Experiment 4 — Benchmark comparison: Wasserstein HMM + MVO vs passive benchmarks.

Benchmarks:
  1. SPX buy-and-hold (fully invested in SPX proxy)
  2. Equal-weight (constant 20% daily rebalanced)

Outputs:
- Sharpe, Sortino, max drawdown comparison table
- Cumulative return and drawdown plots
- HMM weight paths around stress periods
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_prices
from src.features import build_causal_features
from src.metrics import compute_full_metrics
from src.strategy import HMMStrategy
from src.utils import (
    setup_logging, load_manifest, ensure_results_dir,
    plot_cumulative_returns, plot_drawdowns, plot_weights,
    metrics_to_markdown_table, append_to_results_md, save_figure, RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def compute_benchmark_returns(
    returns: pd.DataFrame,
    oos_mask: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Compute SPX buy-and-hold and equal-weight benchmark returns over OOS period.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-returns DataFrame (all assets).
    oos_mask : pd.Series (bool)
        Boolean mask indicating OOS dates.

    Returns
    -------
    spx_returns : pd.Series
        SPX proxy daily log-returns over OOS.
    eq_returns : pd.Series
        Equal-weight portfolio daily log-returns over OOS.
        Convention: constant-mix rebalancing at 1/N each day.
    """
    oos_returns = returns[oos_mask]

    # SPX is the first column (per TICKER_MANIFEST: SPX → SPY)
    spx_returns = oos_returns.iloc[:, 0].rename("SPX_BuyHold")

    # Equal-weight: constant 1/N allocation rebalanced daily
    n = oos_returns.shape[1]
    eq_returns = oos_returns.mean(axis=1).rename("EqualWeight")

    return spx_returns, eq_returns


def run_experiment_4() -> dict:
    """Run benchmark comparison experiment and return metrics."""
    setup_logging()
    ensure_results_dir()

    manifest = load_manifest()
    split_date = manifest["split"]["train_end"]
    mvo_cfg = manifest["mvo"]
    hmm_cfg = manifest["hmm"]
    tmpl_cfg = manifest["templates"]

    logger.info("=" * 60)
    logger.info("Experiment 4: HMM vs SPX Buy-and-Hold vs Equal-Weight")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    prices = load_prices(cache_path="data/prices.parquet")
    returns, features, lagged_features = build_causal_features(prices)
    asset_names = list(prices.columns)

    oos_mask = lagged_features.index > pd.Timestamp(split_date)
    logger.info("OOS: %d trading days", oos_mask.sum())

    # ------------------------------------------------------------------ #
    # 2. HMM strategy — load cached result from exp_1 when available
    # ------------------------------------------------------------------ #
    _cache_path = Path(RESULTS_DIR) / "hmm_result.pkl"
    if _cache_path.exists():
        logger.info("Loading cached HMM result from %s (skipping re-run)", _cache_path)
        with open(_cache_path, "rb") as _f:
            hmm_result = pickle.load(_f)
    else:
        logger.info("Running HMM + Wasserstein template + MVO strategy ...")
        hmm_strategy = HMMStrategy(
            n_templates=tmpl_cfg["n_templates"],
            eta=tmpl_cfg["eta"],
            gamma=mvo_cfg["gamma"],
            tau=mvo_cfg["tau"],
            w_max=mvo_cfg["w_max"],
            k_min=hmm_cfg["k_min"],
            k_max=hmm_cfg["k_max"],
            val_slice_length=hmm_cfg["val_slice_length"],
            select_freq=hmm_cfg["order_select_freq"],
            complexity_penalty=hmm_cfg["complexity_penalty"],
            random_state=hmm_cfg["random_state"],
            n_iter=hmm_cfg["n_iter"],
            max_train_size=hmm_cfg.get("max_train_size", 1000),
            n_assets=len(asset_names),
        )
        hmm_result = hmm_strategy.run(lagged_features, returns, split_date)
    hmm_dfs = hmm_result.to_dataframes(asset_names)

    # ------------------------------------------------------------------ #
    # 3. Passive benchmarks
    # ------------------------------------------------------------------ #
    spx_returns, eq_returns = compute_benchmark_returns(returns, oos_mask)

    # Align to HMM OOS dates
    oos_idx = hmm_dfs["returns"].index
    spx_returns = spx_returns.reindex(oos_idx).fillna(0.0)
    eq_returns = eq_returns.reindex(oos_idx).fillna(0.0)

    logger.info(
        "Benchmarks aligned: SPX BH (%d days), EW (%d days)",
        spx_returns.notna().sum(), eq_returns.notna().sum(),
    )

    # ------------------------------------------------------------------ #
    # 4. Performance metrics
    # ------------------------------------------------------------------ #
    # Build weight DataFrames for benchmarks (for full metrics computation)
    n = len(asset_names)
    spx_weights = pd.DataFrame(
        np.tile([1.0] + [0.0] * (n - 1), (len(oos_idx), 1)),
        index=oos_idx, columns=asset_names,
    )
    eq_weights = pd.DataFrame(
        np.tile([1.0 / n] * n, (len(oos_idx), 1)),
        index=oos_idx, columns=asset_names,
    )

    hmm_metrics = compute_full_metrics(hmm_dfs["returns"], hmm_dfs["weights"], label="HMM+Wasserstein+MVO")
    spx_metrics = compute_full_metrics(spx_returns, spx_weights, label="SPX Buy-and-Hold")
    eq_metrics = compute_full_metrics(eq_returns, eq_weights, label="Equal-Weight (1/N)")

    logger.info("\n--- Performance Summary ---")
    for metrics, label in [(hmm_metrics, "HMM"), (spx_metrics, "SPX BH"), (eq_metrics, "EW")]:
        logger.info(
            "%s | Sharpe=%.4f | MaxDD=%.4f | Sortino=%.4f",
            label, metrics["sharpe"], metrics["max_drawdown"], metrics["sortino"],
        )

    # ------------------------------------------------------------------ #
    # 5. Stress-period analysis (early 2025 selloff)
    # ------------------------------------------------------------------ #
    stress_start = "2025-01-01"
    stress_end = "2025-06-30"
    stress_mask_oos = (oos_idx >= stress_start) & (oos_idx <= stress_end)

    stress_results = {}
    if stress_mask_oos.sum() > 0:
        for label, rets in [
            ("HMM+MVO", hmm_dfs["returns"][stress_mask_oos]),
            ("SPX BH", spx_returns[stress_mask_oos]),
            ("EW", eq_returns[stress_mask_oos]),
        ]:
            cum_ret = float((1 + rets).prod() - 1)
            max_dd = float(((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min())
            stress_results[label] = {"stress_cumret": cum_ret, "stress_maxdd": max_dd}
            logger.info("Stress %s: cum_ret=%.4f, max_dd=%.4f", label, cum_ret, max_dd)

        # HMM weight during stress
        if "weights" in hmm_dfs:
            stress_weights = hmm_dfs["weights"][stress_mask_oos]
            logger.info(
                "Avg HMM weights during stress:\n%s",
                stress_weights.mean().to_string(),
            )
    else:
        logger.warning("No data in stress period %s–%s", stress_start, stress_end)

    # ------------------------------------------------------------------ #
    # 6. Plots
    # ------------------------------------------------------------------ #
    all_rets = {
        "HMM+Wasserstein+MVO": hmm_dfs["returns"],
        "SPX Buy-and-Hold": spx_returns,
        "Equal-Weight (1/N)": eq_returns,
    }

    plot_cumulative_returns(
        all_rets,
        title="Exp 4: Cumulative OOS Returns — HMM vs Passive Benchmarks",
        filename="exp4_cumulative_returns.png",
    )
    plot_drawdowns(
        all_rets,
        title="Exp 4: Drawdown Paths — HMM vs Passive Benchmarks",
        filename="exp4_drawdowns.png",
    )

    # HMM weights plot with stress band
    if "weights" in hmm_dfs:
        fig, ax = plt.subplots(figsize=(14, 5))
        w = hmm_dfs["weights"]
        colors_w = plt.cm.tab10(np.linspace(0, 1, len(asset_names)))
        bottom = np.zeros(len(w))
        for i, asset in enumerate(asset_names):
            ax.bar(w.index, w[asset].values, bottom=bottom,
                   label=asset, color=colors_w[i], alpha=0.85, width=1)
            bottom += w[asset].values
        # Shade stress period
        ax.axvspan(pd.Timestamp(stress_start), pd.Timestamp(stress_end),
                   alpha=0.15, color="red", label="Stress 2025")
        ax.set_title("Exp 4: HMM Portfolio Weights (with Stress Period)", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        save_figure(fig, "exp4_hmm_weights_stress.png")

    # Cumulative return zoom on stress period
    if stress_mask_oos.sum() > 0:
        fig, ax = plt.subplots(figsize=(11, 5))
        colors_s = ["#1f77b4", "#d62728", "#2ca02c"]
        for (label, rets_s), color in zip([
            ("HMM+MVO", hmm_dfs["returns"][stress_mask_oos]),
            ("SPX BH", spx_returns[stress_mask_oos]),
            ("EW", eq_returns[stress_mask_oos]),
        ], colors_s):
            cum = (1 + rets_s).cumprod()
            ax.plot(cum.index, cum.values, label=label, color=color, linewidth=1.5)
        ax.set_title("Exp 4: Stress Period Cumulative Returns (2025)", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.tight_layout()
        save_figure(fig, "exp4_stress_period_returns.png")

    # ------------------------------------------------------------------ #
    # 7. Save results
    # ------------------------------------------------------------------ #
    perf_md = metrics_to_markdown_table(
        [hmm_metrics, eq_metrics, spx_metrics],
        title="Experiment 4 — Performance vs Passive Benchmarks (OOS)",
    )

    content = f"""
## Experiment 4: HMM vs SPX Buy-and-Hold vs Equal-Weight Benchmark

**Train/test split:** `{split_date}`
**OOS period:** {oos_idx[0].date()} → {oos_idx[-1].date()} ({len(oos_idx)} days)

{perf_md}

**Stress Period ({stress_start}–{stress_end}):**
{pd.DataFrame(stress_results).T.to_markdown() if stress_results else "No stress period data available."}

**Key Findings:**
- HMM Sharpe: {hmm_metrics['sharpe']:.4f} | Equal-Weight Sharpe: {eq_metrics['sharpe']:.4f} | SPX BH Sharpe: {spx_metrics['sharpe']:.4f}
- HMM Max Drawdown: {hmm_metrics['max_drawdown']:.4f} | EW: {eq_metrics['max_drawdown']:.4f} | SPX: {spx_metrics['max_drawdown']:.4f}
- Strategy ranking (Sharpe): {"HMM > EW > SPX" if hmm_metrics['sharpe'] > eq_metrics['sharpe'] > spx_metrics['sharpe'] else "See table above"}
"""
    append_to_results_md(content)

    # Save return CSVs
    hmm_dfs["returns"].to_csv(RESULTS_DIR / "exp4_hmm_returns.csv")
    spx_returns.to_csv(RESULTS_DIR / "exp4_spx_returns.csv")
    eq_returns.to_csv(RESULTS_DIR / "exp4_eq_returns.csv")

    logger.info("Experiment 4 complete.")

    return {
        "hmm_metrics": hmm_metrics,
        "spx_metrics": spx_metrics,
        "eq_metrics": eq_metrics,
        "stress_results": stress_results,
        "hmm_dfs": hmm_dfs,
    }


if __name__ == "__main__":
    run_experiment_4()
