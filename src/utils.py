"""Shared utility functions for the Wasserstein HMM replication experiments."""

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_PATH = REPO_ROOT / "config" / "manifest.yaml"


def ensure_results_dir() -> Path:
    """Ensure the results directory exists and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def load_manifest() -> dict:
    """Load the experiment manifest from YAML."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a timestamp-aware handler."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to the results directory."""
    ensure_results_dir()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = "Cumulative Returns",
    filename: str = "cumulative_returns.png",
) -> None:
    """Plot cumulative portfolio returns for multiple strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for (label, rets), color in zip(returns_dict.items(), colors):
        cum = (1 + rets).cumprod()
        ax.plot(cum.index, cum.values, label=label, color=color, linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_weights(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Weights",
    filename: str = "weights.png",
) -> None:
    """Plot portfolio weights over time as a stacked area chart."""
    fig, ax = plt.subplots(figsize=(14, 5))
    assets = weights_df.columns.tolist()
    ax.stackplot(
        weights_df.index,
        [weights_df[a].values for a in assets],
        labels=assets,
        alpha=0.8,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_turnover(
    turnovers_dict: dict[str, pd.Series],
    title: str = "Daily Turnover",
    filename: str = "turnover.png",
) -> None:
    """Plot daily turnover for multiple strategies."""
    fig, axes = plt.subplots(len(turnovers_dict), 1, figsize=(12, 4 * len(turnovers_dict)), sharex=True)
    if len(turnovers_dict) == 1:
        axes = [axes]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, (label, to), color in zip(axes, turnovers_dict.items(), colors):
        ax.bar(to.index, to.values, label=label, color=color, alpha=0.6, width=1)
        ax.set_title(f"{title} — {label}", fontsize=11)
        ax.set_ylabel("Turnover")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    save_figure(fig, filename)


def plot_neff(
    neff_dict: dict[str, pd.Series],
    title: str = "Effective N (Diversification)",
    filename: str = "n_eff.png",
) -> None:
    """Plot effective number of positions over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e"]
    for (label, neff), color in zip(neff_dict.items(), colors):
        ax.plot(neff.index, neff.values, label=label, color=color, alpha=0.8, linewidth=1)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("N_eff")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, filename)


def plot_regime_path(
    regime_labels: pd.Series,
    template_probs: pd.DataFrame,
    n_templates: int = 6,
    title: str = "Regime Path",
    filename: str = "regime_path.png",
) -> None:
    """Plot dominant regime labels and template probabilities."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(regime_labels.index, regime_labels.values, linewidth=0.8, color="#2ca02c")
    ax1.set_ylabel("Dominant Template")
    ax1.set_title(f"{title} — Dominant Regime", fontsize=12)
    ax1.set_yticks(range(n_templates))
    ax1.grid(alpha=0.3)

    colors_t = plt.cm.tab10(np.linspace(0, 1, n_templates))
    for g in range(min(n_templates, template_probs.shape[1])):
        col = template_probs.columns[g]
        ax2.plot(template_probs.index, template_probs[col].values,
                 label=f"T{g}", color=colors_t[g], linewidth=0.8, alpha=0.8)
    ax2.set_ylabel("Template Probability")
    ax2.set_title("Template Probabilities", fontsize=12)
    ax2.legend(ncol=3, fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    save_figure(fig, filename)


def plot_drawdowns(
    returns_dict: dict[str, pd.Series],
    title: str = "Drawdown Paths",
    filename: str = "drawdowns.png",
) -> None:
    """Plot drawdown paths for multiple strategies."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for (label, rets), color in zip(returns_dict.items(), colors):
        cum = (1 + rets).cumprod()
        peak = cum.expanding().max()
        dd = (cum / peak - 1) * 100
        ax.plot(dd.index, dd.values, label=label, color=color, linewidth=1.2)
    ax.fill_between([], [], [], alpha=0)  # dummy for spacing
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def metrics_to_markdown_table(metrics_list: list[dict], title: str = "") -> str:
    """Convert a list of metric dicts to a Markdown table string."""
    if not metrics_list:
        return ""
    keys = [k for k in metrics_list[0].keys() if k != "label"]
    header = "| Strategy | " + " | ".join(k.replace("_", " ").title() for k in keys) + " |"
    sep = "|" + "---|" * (len(keys) + 1)
    rows = []
    for m in metrics_list:
        label = m.get("label", "")
        vals = []
        for k in keys:
            v = m.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append(f"| {label} | " + " | ".join(vals) + " |")
    lines = []
    if title:
        lines.append(f"\n### {title}\n")
    lines += [header, sep] + rows
    return "\n".join(lines)


def append_to_results_md(content: str) -> None:
    """Append content to results/RESULTS.md."""
    ensure_results_dir()
    path = RESULTS_DIR / "RESULTS.md"
    mode = "a" if path.exists() else "w"
    with open(path, mode) as f:
        f.write(content + "\n")
    logger.info("Appended to %s", path)
