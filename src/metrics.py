"""Portfolio performance metrics for quantitative strategy evaluation."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_sharpe(
    returns: np.ndarray | pd.Series,
    annualization: int = 252,
) -> float:
    """Compute the annualized Sharpe ratio (zero risk-free rate assumed).

    Parameters
    ----------
    returns:
        Array of period returns.
    annualization:
        Number of periods per year (252 for daily returns).

    Returns
    -------
    float
        Annualized Sharpe ratio, or ``0.0`` when return volatility is zero.
    """
    r = np.asarray(returns, dtype=float)
    std = r.std()
    if std == 0.0:
        return 0.0
    return float(r.mean() / std * np.sqrt(annualization))


def compute_sortino(
    returns: np.ndarray | pd.Series,
    annualization: int = 252,
) -> float:
    """Compute the annualized Sortino ratio (zero minimum acceptable return).

    Downside volatility is measured using only the returns that fall below
    zero.

    Parameters
    ----------
    returns:
        Array of period returns.
    annualization:
        Number of periods per year.

    Returns
    -------
    float
        Annualized Sortino ratio, or ``0.0`` when downside volatility is zero.
    """
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0.0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    if downside_std == 0.0:
        return 0.0
    return float(r.mean() / downside_std * np.sqrt(annualization))


def compute_max_drawdown(returns: np.ndarray | pd.Series) -> float:
    """Compute the maximum drawdown from a series of period returns.

    Parameters
    ----------
    returns:
        Array of period returns.

    Returns
    -------
    float
        Maximum drawdown as a non-positive fraction (e.g., ``-0.054`` for a
        5.4 % peak-to-trough decline).
    """
    r = np.asarray(returns, dtype=float)
    cum = np.cumprod(1.0 + r)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum / running_max - 1.0
    return float(drawdown.min())


def compute_annualized_return(
    returns: np.ndarray | pd.Series,
    annualization: int = 252,
) -> float:
    """Compute the simple annualized return (arithmetic approximation).

    Parameters
    ----------
    returns:
        Array of period returns.
    annualization:
        Number of periods per year.

    Returns
    -------
    float
        Annualized return = mean(r) * annualization.
    """
    r = np.asarray(returns, dtype=float)
    return float(r.mean() * annualization)


def compute_annualized_vol(
    returns: np.ndarray | pd.Series,
    annualization: int = 252,
) -> float:
    """Compute the annualized volatility.

    Parameters
    ----------
    returns:
        Array of period returns.
    annualization:
        Number of periods per year.

    Returns
    -------
    float
        Annualized volatility = std(r) * sqrt(annualization).
    """
    r = np.asarray(returns, dtype=float)
    return float(r.std() * np.sqrt(annualization))


def compute_turnover_stats(turnovers: np.ndarray | pd.Series) -> dict:
    """Compute descriptive statistics for a turnover series.

    Parameters
    ----------
    turnovers:
        Array of one-way turnover values per period.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``mean``           — average turnover
        * ``median``         — median turnover
        * ``std``            — turnover standard deviation
        * ``p95``            — 95th-percentile turnover
        * ``frac_above_1pct``— fraction of periods with turnover > 1 %
        * ``frac_above_5pct``— fraction of periods with turnover > 5 %
    """
    t = np.asarray(turnovers, dtype=float)
    n = len(t)
    if n == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "frac_above_1pct": 0.0,
            "frac_above_5pct": 0.0,
        }
    return {
        "mean": float(t.mean()),
        "median": float(np.median(t)),
        "std": float(t.std()),
        "p95": float(np.percentile(t, 95)),
        "frac_above_1pct": float((t > 0.01).mean()),
        "frac_above_5pct": float((t > 0.05).mean()),
    }


def compute_effective_n(weights_matrix: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Compute the effective number of assets (inverse-HHI) for each period.

    The effective N at time *t* is defined as:

        N_eff_t = 1 / sum_i(w_{t,i}^2)

    Parameters
    ----------
    weights_matrix:
        Weight matrix of shape ``(T, N)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(T,)`` containing per-period effective N values.
    """
    if isinstance(weights_matrix, pd.DataFrame):
        W = weights_matrix.to_numpy(dtype=float)
    else:
        W = np.asarray(weights_matrix, dtype=float)

    hhi = (W ** 2).sum(axis=1)  # shape (T,)
    # Guard against all-zero weight rows.
    hhi = np.where(hhi == 0.0, np.nan, hhi)
    return 1.0 / hhi


def compute_full_metrics(
    returns: pd.Series,
    weights: pd.DataFrame,
    label: str = "",
) -> dict:
    """Compute the full set of performance metrics for a strategy.

    Turnover is defined as the one-way portfolio churn per period:

        turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|

    Parameters
    ----------
    returns:
        Daily strategy returns as a ``pd.Series``.
    weights:
        Daily portfolio weights as a ``pd.DataFrame`` of shape ``(T, N)``.
        Row order must match ``returns``.
    label:
        Human-readable strategy name embedded in the output dictionary.

    Returns
    -------
    dict
        Dictionary with keys: ``label``, ``sharpe``, ``sortino``,
        ``max_drawdown``, ``ann_return``, ``ann_vol``, ``avg_turnover``,
        ``p95_turnover``, ``frac_above_1pct``, ``frac_above_5pct``,
        ``avg_n_eff``, ``cumulative_return``.
    """
    r = np.asarray(returns, dtype=float)

    # Turnover: first difference of weight matrix (drop the NaN first row).
    weight_diffs = weights.diff().dropna()
    turnovers: np.ndarray = 0.5 * weight_diffs.abs().sum(axis=1).to_numpy(dtype=float)

    turnover_stats = compute_turnover_stats(turnovers)
    n_eff = compute_effective_n(weights)

    cumulative_return = float(np.prod(1.0 + r) - 1.0)

    return {
        "label": label,
        "sharpe": compute_sharpe(r),
        "sortino": compute_sortino(r),
        "max_drawdown": compute_max_drawdown(r),
        "ann_return": compute_annualized_return(r),
        "ann_vol": compute_annualized_vol(r),
        "avg_turnover": turnover_stats["mean"],
        "p95_turnover": turnover_stats["p95"],
        "frac_above_1pct": turnover_stats["frac_above_1pct"],
        "frac_above_5pct": turnover_stats["frac_above_5pct"],
        "avg_n_eff": float(np.nanmean(n_eff)),
        "cumulative_return": cumulative_return,
    }
