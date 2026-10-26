"""
Experiment 1 — Primary replication: Wasserstein HMM vs KNN conditional-moment baseline.

Both strategies feed the same transaction-cost-aware MVO layer. Results include:
- Out-of-sample Sharpe, Sortino, max drawdown
- Turnover statistics (mean, p95, fraction above 1% and 5%)
- Effective diversification (N_eff)
- Weight stability (volatility of weights, absolute weight changes)
- Cumulative return, weight, turnover, N_eff, and regime plots
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

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_prices
from src.features import build_causal_features
from src.metrics import compute_full_metrics, compute_effective_n
from src.strategy import HMMStrategy, KNNStrategy
from src.utils import (
    setup_logging, load_manifest, ensure_results_dir,
    plot_cumulative_returns, plot_weights, plot_turnover, plot_neff,
    plot_regime_path, plot_drawdowns, metrics_to_markdown_table, append_to_results_md,
    save_figure, RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def run_experiment_1() -> dict:
    """Run primary HMM vs KNN replication and return metrics."""
    setup_logging()
    ensure_results_dir()

    manifest = load_manifest()
    split_date = manifest["split"]["train_end"]
    mvo_cfg = manifest["mvo"]
    hmm_cfg = manifest["hmm"]
    knn_cfg = manifest["knn"]
    tmpl_cfg = manifest["templates"]

    logger.info("=" * 60)
    logger.info("Experiment 1: Primary replication — HMM vs KNN")
    logger.info("Train/test split: %s", split_date)
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Data loading and preprocessing
    # ------------------------------------------------------------------ #
    prices = load_prices(cache_path="data/prices.parquet")
    returns, features, lagged_features = build_causal_features(prices)
    asset_names = list(prices.columns)

    logger.info("Data: %d OOS observations, assets: %s", (lagged_features.index > split_date).sum(), asset_names)

    # ------------------------------------------------------------------ #
    # 2. Run HMM strategy
    # ------------------------------------------------------------------ #
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

    # Cache HMM result for exp_4 to avoid a duplicate full run
    _cache_path = Path(RESULTS_DIR) / "hmm_result.pkl"
    with open(_cache_path, "wb") as _f:
        pickle.dump(hmm_result, _f)
    logger.info("HMM result cached to %s", _cache_path)

    hmm_dfs = hmm_result.to_dataframes(asset_names)

    # ------------------------------------------------------------------ #
    # 3. Run KNN strategy
    # ------------------------------------------------------------------ #
    logger.info("Running KNN conditional-moment + MVO strategy ...")
    knn_strategy = KNNStrategy(
        k_neighbors=knn_cfg["k_neighbors"],
        gamma=mvo_cfg["gamma"],
        tau=mvo_cfg["tau"],
        w_max=mvo_cfg["w_max"],
        n_assets=len(asset_names),
        metric=knn_cfg["metric"],
    )
    knn_result = knn_strategy.run(lagged_features, returns, split_date)
    knn_dfs = knn_result.to_dataframes(asset_names)

    # ------------------------------------------------------------------ #
    # 4. Compute metrics
    # ------------------------------------------------------------------ #
    hmm_metrics = compute_full_metrics(
        hmm_dfs["returns"], hmm_dfs["weights"], label="HMM+Wasserstein+MVO"
    )
    knn_metrics = compute_full_metrics(
        knn_dfs["returns"], knn_dfs["weights"], label="KNN+MVO"
    )

    logger.info("\n--- HMM Metrics ---")
    for k, v in hmm_metrics.items():
        if k != "label":
            logger.info("  %-25s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    logger.info("\n--- KNN Metrics ---")
    for k, v in knn_metrics.items():
        if k != "label":
            logger.info("  %-25s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    # ------------------------------------------------------------------ #
    # 5. Weight stability supplementary metrics
    # ------------------------------------------------------------------ #
    def weight_stability_metrics(weights_df: pd.DataFrame, label: str) -> dict:
        dw = weights_df.diff().dropna()
        avg_weight = weights_df.mean().to_dict()
        weight_vol = weights_df.std().to_dict()
        avg_abs_change = dw.abs().mean().to_dict()
        return {
            "label": label,
            "avg_weight_spx": avg_weight.get(asset_names[0], np.nan),
            "avg_weight_bond": avg_weight.get(asset_names[1], np.nan),
            "avg_weight_gold": avg_weight.get(asset_names[2], np.nan),
            "avg_weight_oil": avg_weight.get(asset_names[3], np.nan),
            "avg_weight_usd": avg_weight.get(asset_names[4], np.nan),
            "weight_vol_spx": weight_vol.get(asset_names[0], np.nan),
            "avg_abs_change_spx": avg_abs_change.get(asset_names[0], np.nan),
        }

    hmm_stab = weight_stability_metrics(hmm_dfs["weights"], "HMM+Wasserstein+MVO")
    knn_stab = weight_stability_metrics(knn_dfs["weights"], "KNN+MVO")

    # ------------------------------------------------------------------ #
    # 6. Regime-level summary
    # ------------------------------------------------------------------ #
    regime_summary = {}
    if "dominant_template" in hmm_dfs and "template_probs" in hmm_dfs:
        dom = hmm_dfs["dominant_template"]
        tp = hmm_dfs["template_probs"]
        occupancy = dom.value_counts().sort_index()
        regime_summary["template_occupancy"] = occupancy.to_dict()
        dormant_g = tp.mean().idxmin()
        regime_summary["dormant_template"] = dormant_g
        regime_summary["min_avg_template_prob"] = float(tp.mean().min())
        logger.info("Dominant template occupancy:\n%s", occupancy)
        logger.info("Likely dormant template: %s (avg prob=%.4f)", dormant_g, tp.mean().min())

    # ------------------------------------------------------------------ #
    # 7. Plots
    # ------------------------------------------------------------------ #
    logger.info("Generating plots ...")

    # Cumulative returns
    plot_cumulative_returns(
        {"HMM+Wasserstein+MVO": hmm_dfs["returns"], "KNN+MVO": knn_dfs["returns"]},
        title="Exp 1: Cumulative OOS Returns — HMM vs KNN",
        filename="exp1_cumulative_returns.png",
    )

    # Drawdowns
    plot_drawdowns(
        {"HMM+Wasserstein+MVO": hmm_dfs["returns"], "KNN+MVO": knn_dfs["returns"]},
        title="Exp 1: Drawdown Paths — HMM vs KNN",
        filename="exp1_drawdowns.png",
    )

    # Weights
    plot_weights(hmm_dfs["weights"], title="HMM Strategy Weights", filename="exp1_hmm_weights.png")
    plot_weights(knn_dfs["weights"], title="KNN Strategy Weights", filename="exp1_knn_weights.png")

    # Turnover
    plot_turnover(
        {"HMM+Wasserstein+MVO": hmm_dfs["turnovers"], "KNN+MVO": knn_dfs["turnovers"]},
        title="Exp 1: Daily Turnover",
        filename="exp1_turnover.png",
    )

    # N_eff
    plot_neff(
        {"HMM+Wasserstein+MVO": hmm_dfs["n_eff"], "KNN+MVO": knn_dfs["n_eff"]},
        title="Exp 1: Effective N (Diversification)",
        filename="exp1_n_eff.png",
    )

    # Regime path
    if "dominant_template" in hmm_dfs and "template_probs" in hmm_dfs:
        plot_regime_path(
            hmm_dfs["dominant_template"],
            hmm_dfs["template_probs"],
            n_templates=tmpl_cfg["n_templates"],
            title="Exp 1: HMM Regime Path",
            filename="exp1_regime_path.png",
        )

    # Selected K over time
    if "selected_k" in hmm_dfs:
        fig, ax = plt.subplots(figsize=(12, 4))
        sk = hmm_dfs["selected_k"]
        ax.step(sk.index, sk.values, where="post", color="#2ca02c", linewidth=1.2)
        ax.set_title("Exp 1: Adaptive Selected K over OOS Period", fontsize=12)
        ax.set_ylabel("Selected K")
        ax.set_xlabel("Date")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_figure(fig, "exp1_selected_k.png")

    # ------------------------------------------------------------------ #
    # 8. Save results to RESULTS.md
    # ------------------------------------------------------------------ #
    performance_md = metrics_to_markdown_table(
        [hmm_metrics, knn_metrics],
        title="Experiment 1 — Performance Metrics (OOS)",
    )

    stability_md = metrics_to_markdown_table(
        [hmm_stab, knn_stab],
        title="Experiment 1 — Weight Stability",
    )

    content = f"""
## Experiment 1: Primary Replication — Wasserstein HMM vs KNN

**Train/test split:** `{split_date}`
**Asset universe:** {asset_names}
**OOS period:** {hmm_dfs['returns'].index[0].date()} → {hmm_dfs['returns'].index[-1].date()}

{performance_md}

{stability_md}

**Regime Summary:**
- Template occupancy: {regime_summary.get('template_occupancy', {})}
- Likely dormant template: {regime_summary.get('dormant_template', 'N/A')} (avg prob = {regime_summary.get('min_avg_template_prob', float('nan')):.4f})

**Key Findings:**
- HMM Sharpe: {hmm_metrics['sharpe']:.4f} vs KNN Sharpe: {knn_metrics['sharpe']:.4f}
- HMM Max Drawdown: {hmm_metrics['max_drawdown']:.4f} vs KNN Max Drawdown: {knn_metrics['max_drawdown']:.4f}
- HMM Avg Turnover: {hmm_metrics['avg_turnover']:.6f} vs KNN Avg Turnover: {knn_metrics['avg_turnover']:.6f}
- HMM Avg N_eff: {hmm_metrics['avg_n_eff']:.4f} vs KNN Avg N_eff: {knn_metrics['avg_n_eff']:.4f}
"""
    append_to_results_md(content)

    # Save CSV summaries
    hmm_dfs["returns"].to_csv(RESULTS_DIR / "exp1_hmm_returns.csv")
    knn_dfs["returns"].to_csv(RESULTS_DIR / "exp1_knn_returns.csv")
    hmm_dfs["weights"].to_csv(RESULTS_DIR / "exp1_hmm_weights.csv")
    knn_dfs["weights"].to_csv(RESULTS_DIR / "exp1_knn_weights.csv")

    logger.info("Experiment 1 complete.")

    return {
        "hmm_metrics": hmm_metrics,
        "knn_metrics": knn_metrics,
        "hmm_dfs": hmm_dfs,
        "knn_dfs": knn_dfs,
        "regime_summary": regime_summary,
    }


if __name__ == "__main__":
    run_experiment_1()
