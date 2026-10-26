"""
Experiment 2 — Adaptive HMM regime complexity selection.

Tests the paper's claim that time-varying HMM state count selected by
penalized one-step-ahead predictive log-likelihood is preferable to a fixed
regime count. This experiment is self-contained and does not require portfolio
optimization.

Outputs:
- Daily (carried-forward) K_t path
- Predictive log-likelihood scores by candidate K
- Comparison of adaptive vs. fixed-K cumulative predictive LL
- CSV audit trail with per-date scores
"""

import logging
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
from src.hmm_model import (
    fit_gaussian_hmm, compute_predictive_loglik,
    K_MIN, K_MAX, VAL_SLICE_LENGTH, ORDER_SELECT_FREQ, COMPLEXITY_PENALTY,
)
from src.utils import (
    setup_logging, load_manifest, ensure_results_dir,
    append_to_results_md, save_figure, RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def run_experiment_2() -> dict:
    """Run adaptive HMM order selection experiment and return results."""
    setup_logging()
    ensure_results_dir()

    manifest = load_manifest()
    split_date = manifest["split"]["train_end"]
    hmm_cfg = manifest["hmm"]

    k_min = hmm_cfg["k_min"]
    k_max = hmm_cfg["k_max"]
    val_slice = hmm_cfg["val_slice_length"]
    select_freq = hmm_cfg["order_select_freq"]
    penalty = hmm_cfg["complexity_penalty"]
    random_state = hmm_cfg["random_state"]
    n_iter = hmm_cfg["n_iter"]
    max_train_size = hmm_cfg.get("max_train_size", 1000)

    logger.info("=" * 60)
    logger.info("Experiment 2: Adaptive HMM Order Selection")
    logger.info("K range: [%d, %d], val_slice=%d, freq=%d, penalty=%.2f",
                k_min, k_max, val_slice, select_freq, penalty)
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Data preparation
    # ------------------------------------------------------------------ #
    prices = load_prices(cache_path="data/prices.parquet")
    returns, features, lagged_features = build_causal_features(prices)

    X_all = lagged_features.values.astype(float)
    dates_all = lagged_features.index

    train_mask = dates_all <= pd.Timestamp(split_date)
    oos_mask = ~train_mask

    train_idx = np.where(train_mask)[0]
    oos_idx = np.where(oos_mask)[0]

    logger.info(
        "OOS: %d days (%s → %s)",
        len(oos_idx), dates_all[oos_idx[0]].date(), dates_all[oos_idx[-1]].date()
    )

    # ------------------------------------------------------------------ #
    # 2. Order selection loop over OOS period
    # ------------------------------------------------------------------ #
    audit_records = []
    daily_k = []
    current_k = k_min  # carry-forward initialized to k_min

    for step_i, t in enumerate(oos_idx):
        date = dates_all[t]
        X_hist = X_all[max(0, t - max_train_size):t]  # capped expanding window up to t-1

        # Only run selection on scheduled dates
        if step_i % select_freq != 0 or len(X_hist) < val_slice + k_min + 5:
            daily_k.append((date, current_k, None))
            continue

        # Split: train = H_t minus validation tail, val = last val_slice obs
        n_hist = len(X_hist)
        val_start = n_hist - val_slice
        if val_start < k_min + 5:
            daily_k.append((date, current_k, None))
            continue

        X_train_sel = X_hist[:val_start]
        X_val_sel = X_hist[val_start:]

        best_k = current_k
        best_score = -np.inf
        candidate_scores = {}

        for k in range(k_min, k_max + 1):
            model = fit_gaussian_hmm(X_train_sel, k, random_state=random_state, n_iter=n_iter)
            if model is None:
                candidate_scores[k] = {"pred_ll": np.nan, "penalty": k * penalty, "penalized": np.nan}
                audit_records.append({
                    "selection_date": date,
                    "candidate_K": k,
                    "pred_ll": np.nan,
                    "complexity_penalty": k * penalty,
                    "penalized_score": np.nan,
                    "converged": False,
                    "selected_flag": False,
                })
                continue

            pred_ll = compute_predictive_loglik(model, X_val_sel)
            pen_score = pred_ll - penalty * k

            candidate_scores[k] = {"pred_ll": pred_ll, "penalty": k * penalty, "penalized": pen_score}
            audit_records.append({
                "selection_date": date,
                "candidate_K": k,
                "pred_ll": pred_ll,
                "complexity_penalty": k * penalty,
                "penalized_score": pen_score,
                "converged": True,
                "selected_flag": False,
            })

            if pen_score > best_score:
                best_score = pen_score
                best_k = k

        # Mark selected K
        for rec in audit_records:
            if rec["selection_date"] == date and rec["candidate_K"] == best_k:
                rec["selected_flag"] = True

        current_k = best_k
        daily_k.append((date, current_k, candidate_scores))

        if step_i % 20 == 0:
            logger.info(
                "Selection step %d/%d | date=%s | selected K=%d | best penalized LL=%.2f",
                step_i + 1, len(oos_idx), date.date(), current_k, best_score,
            )

    # Fill non-selection dates with carry-forward K
    k_path_dates = [r[0] for r in daily_k]
    k_path_vals = [r[1] for r in daily_k]
    k_series = pd.Series(k_path_vals, index=pd.DatetimeIndex(k_path_dates), name="selected_K")

    # Fill all OOS dates (including non-selection ones) with carry-forward
    all_oos_dates = dates_all[oos_idx]
    k_series = k_series.reindex(all_oos_dates, method="ffill")

    # ------------------------------------------------------------------ #
    # 3. Compute fixed-K cumulative predictive LL for comparison
    # ------------------------------------------------------------------ #
    audit_df = pd.DataFrame(audit_records)

    # Per-K aggregated predictive LL across all selection dates
    if not audit_df.empty and "pred_ll" in audit_df.columns:
        k_scores_agg = (
            audit_df.dropna(subset=["pred_ll"])
            .groupby("candidate_K")["pred_ll"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "avg_pred_ll", "sum": "total_pred_ll", "count": "n_obs"})
        )
        logger.info("Per-K aggregate predictive LL:\n%s", k_scores_agg.to_string())
    else:
        k_scores_agg = pd.DataFrame()

    # Adaptive K aggregate: use selected K's pred_ll per date
    if not audit_df.empty:
        selected_df = audit_df[audit_df["selected_flag"]]
        adaptive_total_ll = selected_df["pred_ll"].sum()
        adaptive_avg_ll = selected_df["pred_ll"].mean()
        logger.info("Adaptive K total pred LL: %.4f, avg: %.4f", adaptive_total_ll, adaptive_avg_ll)
    else:
        adaptive_total_ll = adaptive_avg_ll = float("nan")

    # ------------------------------------------------------------------ #
    # 4. K_t distribution statistics
    # ------------------------------------------------------------------ #
    k_freq = k_series.value_counts().sort_index()
    k_transitions = int((k_series.diff() != 0).sum())

    logger.info("K_t distribution:\n%s", k_freq.to_string())
    logger.info("K_t transitions (changes): %d", k_transitions)
    logger.info("K_t range: [%d, %d]", int(k_series.min()), int(k_series.max()))

    # ------------------------------------------------------------------ #
    # 5. Plots
    # ------------------------------------------------------------------ #
    # K_t path over time
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(k_series.index, k_series.values, where="post", color="#1f77b4", linewidth=1.5, label="Adaptive K")
    ax.axhline(y=k_series.mean(), color="gray", linestyle="--", label=f"Mean K = {k_series.mean():.2f}")
    ax.set_title("Exp 2: Adaptive Selected K_t over OOS Period", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Selected K")
    ax.set_yticks(range(k_min, k_max + 1))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    save_figure(fig, "exp2_k_path.png")

    # K distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    k_freq.plot(kind="bar", ax=ax, color="#ff7f0e", edgecolor="black")
    ax.set_title("Exp 2: Frequency Distribution of Selected K", fontsize=12)
    ax.set_xlabel("K")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure(fig, "exp2_k_distribution.png")

    # Per-K cumulative predictive LL (if available)
    if not k_scores_agg.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = k_scores_agg.index.astype(str)
        ax.bar(x, k_scores_agg["avg_pred_ll"], color="#2ca02c", edgecolor="black", label="Fixed K avg pred LL")
        if not np.isnan(adaptive_avg_ll):
            ax.axhline(y=adaptive_avg_ll, color="red", linestyle="--",
                       label=f"Adaptive K avg pred LL = {adaptive_avg_ll:.2f}")
        ax.set_title("Exp 2: Average Predictive LL by Candidate K", fontsize=12)
        ax.set_xlabel("K")
        ax.set_ylabel("Avg Predictive Log-Likelihood")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        save_figure(fig, "exp2_pred_ll_by_k.png")

    # ------------------------------------------------------------------ #
    # 6. Save audit trail and results
    # ------------------------------------------------------------------ #
    if not audit_df.empty:
        audit_df.to_csv(RESULTS_DIR / "exp2_order_selection_audit.csv", index=False)
        logger.info("Saved audit trail: exp2_order_selection_audit.csv")

    k_series.to_csv(RESULTS_DIR / "exp2_k_path.csv", header=True)

    content = f"""
## Experiment 2: Adaptive HMM Order Selection

**Config:** K range [{k_min}, {k_max}], val_slice={val_slice}, select_freq={select_freq}, penalty={penalty}

**K_t Path Statistics:**
| Metric | Value |
|--------|-------|
| Min K | {int(k_series.min())} |
| Max K | {int(k_series.max())} |
| Mean K | {k_series.mean():.2f} |
| Std K | {k_series.std():.2f} |
| Num Transitions | {k_transitions} |

**K Frequency Distribution:**
{k_freq.to_markdown()}

**Predictive LL Comparison (selection dates):**
| Method | Avg Pred LL | Total Pred LL |
|--------|-------------|---------------|
| Adaptive K | {adaptive_avg_ll:.4f} | {adaptive_total_ll:.4f} |
{chr(10).join([f"| Fixed K={row.name} | {row['avg_pred_ll']:.4f} | {row['total_pred_ll']:.4f} |" for _, row in k_scores_agg.iterrows()]) if not k_scores_agg.empty else ""}

**Key Finding:** The adaptive scheme produces K_t = {int(k_series.mode()[0])} most frequently.
"""
    append_to_results_md(content)
    logger.info("Experiment 2 complete.")

    return {
        "k_series": k_series,
        "audit_df": audit_df,
        "k_scores_agg": k_scores_agg,
        "adaptive_total_ll": adaptive_total_ll,
        "adaptive_avg_ll": adaptive_avg_ll,
        "k_transitions": k_transitions,
    }


if __name__ == "__main__":
    run_experiment_2()
