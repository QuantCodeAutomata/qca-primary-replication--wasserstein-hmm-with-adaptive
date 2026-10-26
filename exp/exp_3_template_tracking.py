"""
Experiment 3 — Wasserstein template-based regime identity tracking.

Tests whether persistent template mapping via 2-Wasserstein distance
stabilizes regime identity across repeated HMM re-estimation.

Outputs:
- Daily dominant template labels and template probabilities
- Template occupancy and stability statistics
- Template interpretability (return-block means per template)
- Comparison: template-mapped labels vs naive raw HMM labels
- Plots of template trajectories, occupancy, and stress-period behavior
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
    fit_gaussian_hmm, get_predictive_state_probs,
    extract_hmm_components, K_MIN,
)
from src.wasserstein import WassersteinTemplateTracker, gaussian_w2_squared
from src.utils import (
    setup_logging, load_manifest, ensure_results_dir,
    append_to_results_md, save_figure, plot_regime_path, RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def compute_label_stability(labels: pd.Series) -> dict:
    """Compute persistence and switching statistics for a label series."""
    changes = (labels.diff() != 0).sum()
    persistence = 1.0 - float(changes) / max(len(labels) - 1, 1)
    entropy_vals = labels.value_counts(normalize=True)
    entropy = float(-np.sum(entropy_vals * np.log(entropy_vals + 1e-12)))
    return {
        "n_changes": int(changes),
        "persistence": float(persistence),
        "entropy": float(entropy),
        "unique_labels": int(labels.nunique()),
    }


def run_experiment_3() -> dict:
    """Run template tracking experiment and return results."""
    setup_logging()
    ensure_results_dir()

    manifest = load_manifest()
    split_date = manifest["split"]["train_end"]
    hmm_cfg = manifest["hmm"]
    tmpl_cfg = manifest["templates"]

    n_templates = tmpl_cfg["n_templates"]
    eta = tmpl_cfg["eta"]
    random_state = hmm_cfg["random_state"]
    n_iter = hmm_cfg["n_iter"]
    max_train_size = hmm_cfg.get("max_train_size", 1000)
    n_assets = 5
    D = 3 * n_assets  # feature dimension

    # Fixed K for this experiment (use mean of range for stability)
    fixed_k = (hmm_cfg["k_min"] + hmm_cfg["k_max"]) // 2
    logger.info("=" * 60)
    logger.info("Experiment 3: Wasserstein Template Tracking (G=%d, fixed K=%d)", n_templates, fixed_k)
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Data preparation
    # ------------------------------------------------------------------ #
    prices = load_prices(cache_path="data/prices.parquet")
    returns, features, lagged_features = build_causal_features(prices)
    asset_names = list(prices.columns)

    X_all = lagged_features.values.astype(float)
    dates_all = lagged_features.index

    train_mask = dates_all <= pd.Timestamp(split_date)
    oos_mask = ~train_mask
    train_idx = np.where(train_mask)[0]
    oos_idx = np.where(oos_mask)[0]

    logger.info(
        "OOS: %d days (%s → %s)",
        len(oos_idx), dates_all[oos_idx[0]].date(), dates_all[oos_idx[-1]].date(),
    )

    # ------------------------------------------------------------------ #
    # 2. Initialize templates from training data
    # ------------------------------------------------------------------ #
    tracker = WassersteinTemplateTracker(
        n_templates=n_templates, eta=eta, feature_dim=D, n_assets=n_assets
    )

    X_train = X_all[train_idx]
    init_model = fit_gaussian_hmm(X_train, min(n_templates, hmm_cfg["k_max"]), random_state=random_state, n_iter=n_iter)
    if init_model is not None:
        means_init, covars_init = extract_hmm_components(init_model)
        tracker.initialize_from_hmm(means_init, covars_init)
        logger.info("Templates initialized from K=%d HMM on %d training obs.", min(n_templates, hmm_cfg["k_max"]), len(X_train))
    else:
        logger.warning("Template initialization HMM failed; using defaults.")

    # ------------------------------------------------------------------ #
    # 3. Main loop: daily expanding-window HMM + template tracking
    # ------------------------------------------------------------------ #
    # Template tracking records
    dominant_labels = []    # template-mapped dominant labels
    template_probs_list = []
    raw_hmm_labels = []     # naive raw component argmax (for comparison)
    wass_drifts = []        # drift of dominant template per step
    assignments_log = []    # component-to-template assignments

    prev_dom = 0

    for step_i, t in enumerate(oos_idx):
        date = dates_all[t]
        X_hist = X_all[max(0, t - max_train_size):t]  # capped expanding window

        if len(X_hist) < fixed_k + 10:
            dominant_labels.append(0)
            template_probs_list.append(np.ones(n_templates) / n_templates)
            raw_hmm_labels.append(0)
            wass_drifts.append(0.0)
            assignments_log.append(np.zeros(fixed_k, dtype=int))
            continue

        # Fit HMM on expanding history (fixed K for this experiment)
        model = fit_gaussian_hmm(X_hist, fixed_k, random_state=random_state, n_iter=n_iter)
        if model is None:
            dominant_labels.append(prev_dom)
            template_probs_list.append(np.ones(n_templates) / n_templates)
            raw_hmm_labels.append(0)
            wass_drifts.append(0.0)
            assignments_log.append(np.zeros(fixed_k, dtype=int))
            continue

        # One-step-ahead predictive state probs
        state_probs = get_predictive_state_probs(model, X_hist)
        comp_means, comp_covars = extract_hmm_components(model)

        # Wasserstein template step
        tmpl_probs, assignments, mu_ret, Sigma_ret = tracker.step(
            comp_means, comp_covars, state_probs
        )

        # Dominant template (template-mapped)
        dom = int(np.argmax(tmpl_probs))
        dominant_labels.append(dom)
        template_probs_list.append(tmpl_probs.copy())

        # Naive raw label: argmax of state_probs (raw component index)
        raw_dom = int(np.argmax(state_probs))
        raw_hmm_labels.append(raw_dom)

        # Wasserstein drift of the dominant template
        if step_i > 0 and prev_dom == dom:
            # Compute W2 from dominant template to itself across two consecutive steps
            # (approximate: measure change in template params)
            drift = 0.0  # self-distance is 0 by definition; track inter-step param change
        else:
            drift = 1.0  # template switch

        wass_drifts.append(drift)
        assignments_log.append(assignments.copy())
        prev_dom = dom

        if step_i % 50 == 0:
            logger.info(
                "Template step %d/%d | date=%s | dominant=%d | raw=%d | tmpl_probs=%s",
                step_i + 1, len(oos_idx), date.date(), dom, raw_dom,
                np.round(tmpl_probs, 3),
            )

    # ------------------------------------------------------------------ #
    # 4. Stability comparison: template labels vs raw HMM labels
    # ------------------------------------------------------------------ #
    oos_dates = pd.DatetimeIndex(dates_all[oos_idx])
    dom_series = pd.Series(dominant_labels, index=oos_dates, name="dominant_template")
    raw_series = pd.Series(raw_hmm_labels, index=oos_dates, name="raw_hmm_label")
    tp_df = pd.DataFrame(
        template_probs_list, index=oos_dates,
        columns=[f"Template_{g}" for g in range(n_templates)],
    )

    template_stability = compute_label_stability(dom_series)
    raw_stability = compute_label_stability(raw_series)

    logger.info("Template label stability: %s", template_stability)
    logger.info("Raw HMM label stability:  %s", raw_stability)

    # Template occupancy
    occupancy = dom_series.value_counts().sort_index()
    logger.info("Template occupancy:\n%s", occupancy.to_string())

    # Average template probabilities (to identify dormant template)
    avg_tp = tp_df.mean()
    dormant_template = int(avg_tp.idxmin().split("_")[-1])
    logger.info(
        "Average template probs:\n%s\nLikely dormant: Template_%d (avg prob=%.4f)",
        avg_tp.to_string(), dormant_template, float(avg_tp.min()),
    )

    # ------------------------------------------------------------------ #
    # 5. Template interpretability: return-block means per template
    # ------------------------------------------------------------------ #
    template_return_means = {}
    for g in range(n_templates):
        mu_g_full = tracker.template_means[g]   # shape (D,)
        # Return block: first n_assets elements correspond to asset returns
        mu_g_ret = mu_g_full[:n_assets]
        template_return_means[f"Template_{g}"] = {
            asset: float(mu_g_ret[i]) for i, asset in enumerate(asset_names)
        }

    tmpl_interp_df = pd.DataFrame(template_return_means).T
    logger.info("Template return-block means:\n%s", tmpl_interp_df.to_string())

    # ------------------------------------------------------------------ #
    # 6. Stress-period analysis (early 2025 selloff)
    # ------------------------------------------------------------------ #
    stress_start = "2025-01-01"
    stress_end = "2025-06-30"
    stress_mask = (dom_series.index >= stress_start) & (dom_series.index <= stress_end)
    if stress_mask.sum() > 0:
        stress_dom = dom_series[stress_mask]
        stress_tp = tp_df[stress_mask]
        stress_occupancy = stress_dom.value_counts().sort_index()
        logger.info("Stress period (%s–%s) template occupancy:\n%s",
                    stress_start, stress_end, stress_occupancy.to_string())
    else:
        logger.info("No data in stress period %s–%s.", stress_start, stress_end)
        stress_occupancy = pd.Series(dtype=int)

    # ------------------------------------------------------------------ #
    # 7. Plots
    # ------------------------------------------------------------------ #
    plot_regime_path(
        dom_series, tp_df,
        n_templates=n_templates,
        title="Exp 3: Wasserstein Template Regime Path",
        filename="exp3_template_path.png",
    )

    # Template vs raw label comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax1.plot(dom_series.index, dom_series.values, linewidth=0.8, color="#1f77b4")
    ax1.set_ylabel("Template Label (Wasserstein)")
    ax1.set_title("Exp 3: Template-Mapped vs Raw HMM Labels", fontsize=12)
    ax1.set_yticks(range(n_templates))
    ax1.grid(alpha=0.3)
    ax2.plot(raw_series.index, raw_series.values, linewidth=0.8, color="#ff7f0e", alpha=0.8)
    ax2.set_ylabel(f"Raw HMM Component Label (K={fixed_k})")
    ax2.set_yticks(range(fixed_k))
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    save_figure(fig, "exp3_label_comparison.png")

    # Template return-block heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(tmpl_interp_df.values, aspect="auto", cmap="RdYlGn", vmin=-0.001, vmax=0.001)
    ax.set_xticks(range(len(asset_names)))
    ax.set_xticklabels(asset_names)
    ax.set_yticks(range(n_templates))
    ax.set_yticklabels([f"Template_{g}" for g in range(n_templates)])
    ax.set_title("Exp 3: Template Return-Block Means", fontsize=12)
    plt.colorbar(im, ax=ax, label="Mean Return")
    fig.tight_layout()
    save_figure(fig, "exp3_template_return_heatmap.png")

    # Occupancy bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    occupancy.plot(kind="bar", ax=ax, color="#2ca02c", edgecolor="black")
    ax.set_title("Exp 3: Template Occupancy (OOS Period)", fontsize=12)
    ax.set_xlabel("Template")
    ax.set_ylabel("Days")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save_figure(fig, "exp3_template_occupancy.png")

    # ------------------------------------------------------------------ #
    # 8. Save results
    # ------------------------------------------------------------------ #
    dom_series.to_csv(RESULTS_DIR / "exp3_dominant_template.csv", header=True)
    tp_df.to_csv(RESULTS_DIR / "exp3_template_probs.csv")
    tmpl_interp_df.to_csv(RESULTS_DIR / "exp3_template_return_means.csv")

    content = f"""
## Experiment 3: Wasserstein Template Tracking

**Config:** G={n_templates}, eta={eta}, fixed K={fixed_k}

**Label Stability Comparison:**
| Metric | Template-Mapped | Raw HMM |
|--------|----------------|---------|
| N changes | {template_stability['n_changes']} | {raw_stability['n_changes']} |
| Persistence | {template_stability['persistence']:.4f} | {raw_stability['persistence']:.4f} |
| Entropy | {template_stability['entropy']:.4f} | {raw_stability['entropy']:.4f} |
| Unique Labels | {template_stability['unique_labels']} | {raw_stability['unique_labels']} |

**Template Occupancy:**
{occupancy.to_markdown()}

**Dominant Template Avg Probs:**
{avg_tp.to_markdown()}

**Likely Dormant Template:** Template_{dormant_template} (avg prob = {float(avg_tp.min()):.4f})

**Template Return-Block Means:**
{tmpl_interp_df.to_markdown()}

**Stress Period ({stress_start}–{stress_end}) Occupancy:**
{stress_occupancy.to_markdown() if len(stress_occupancy) > 0 else "No data"}
"""
    append_to_results_md(content)
    logger.info("Experiment 3 complete.")

    return {
        "dom_series": dom_series,
        "raw_series": raw_series,
        "tp_df": tp_df,
        "template_stability": template_stability,
        "raw_stability": raw_stability,
        "occupancy": occupancy,
        "tmpl_interp_df": tmpl_interp_df,
        "dormant_template": dormant_template,
    }


if __name__ == "__main__":
    run_experiment_3()
