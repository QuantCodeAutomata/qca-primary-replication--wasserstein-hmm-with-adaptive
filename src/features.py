"""src/features.py — Strictly causal feature engineering for return series.

Pipeline
--------
prices  ──►  log_returns  ──►  (rolling_vol, rolling_mean)
                                        │
                               build_feature_matrix
                                        │
                               features  ──shift(1)──►  lagged_features

Public entry point: :func:`build_causal_features` returns the triple
(returns, features, lagged_features) aligned to a shared date index so
that ``lagged_features.iloc[t]`` contains only information available
strictly before date ``t``.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ROLL_VOL_WINDOW: int = 60   # lookback for rolling standard deviation
ROLL_MEAN_WINDOW: int = 20  # lookback for rolling mean (momentum proxy)
N_ASSETS: int = 5


# ---------------------------------------------------------------------------
# Step 1 – log returns
# ---------------------------------------------------------------------------


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame.

    .. math::
        r_t = \\ln\\!\\left(\\frac{P_t}{P_{t-1}}\\right)

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed price DataFrame with one column per asset.

    Returns
    -------
    pd.DataFrame
        Log-return DataFrame with the same column layout as *prices*.
        The first row (which would be NaN) is dropped, so the returned
        index starts one period after ``prices.index[0]``.
    """
    log_returns: pd.DataFrame = np.log(prices / prices.shift(1)).dropna()
    logger.debug(
        "compute_log_returns: input rows=%d → output rows=%d",
        len(prices),
        len(log_returns),
    )
    return log_returns


# ---------------------------------------------------------------------------
# Step 2 – rolling statistics (unshifted)
# ---------------------------------------------------------------------------


def compute_rolling_features(
    returns: pd.DataFrame,
    vol_window: int = ROLL_VOL_WINDOW,
    mean_window: int = ROLL_MEAN_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling volatility and rolling mean of log returns.

    .. note::
        **No shifting is applied here.**  Both outputs are same-day
        estimates (e.g. ``rolling_vol.iloc[t]`` uses returns from
        ``t - vol_window + 1`` through ``t`` inclusive).  The caller is
        responsible for lagging the results to preserve causality.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-return series with one column per asset.
    vol_window : int
        Number of periods for the rolling standard-deviation window.
        Defaults to :data:`ROLL_VOL_WINDOW` (60).
    mean_window : int
        Number of periods for the rolling mean window.
        Defaults to :data:`ROLL_MEAN_WINDOW` (20).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(rolling_vol, rolling_mean)`` — both DataFrames share the index
        and column layout of *returns*, with ``NaN`` during the warm-up
        period.
    """
    rolling_vol: pd.DataFrame = returns.rolling(vol_window).std()
    rolling_mean: pd.DataFrame = returns.rolling(mean_window).mean()
    logger.debug(
        "compute_rolling_features: vol_window=%d, mean_window=%d",
        vol_window,
        mean_window,
    )
    return rolling_vol, rolling_mean


# ---------------------------------------------------------------------------
# Step 3 – feature matrix assembly
# ---------------------------------------------------------------------------


def build_feature_matrix(
    returns: pd.DataFrame,
    rolling_vol: pd.DataFrame,
    rolling_mean: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble a flat feature matrix from returns and rolling statistics.

    Columns are stacked by feature type to match the paper's notation
    x_t = [r_t ; σ_t ; m_t] ∈ R^{3N}:

    ``[SPX_ret, BOND_ret, …, SPX_vol, BOND_vol, …, SPX_mom, BOND_mom, …]``

    This ordering ensures that ``feature_matrix.values[:, :N]`` yields the
    pure return block, ``[:, N:2N]`` the volatility block, and
    ``[:, 2N:]`` the momentum block — required by the Wasserstein template
    extractor which reads the return-block as ``means[:N]``.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-return series (asset names as columns).
    rolling_vol : pd.DataFrame
        Rolling standard deviation aligned to *returns*.
    rolling_mean : pd.DataFrame
        Rolling mean aligned to *returns*.

    Returns
    -------
    pd.DataFrame
        Feature matrix with ``3 × N_assets`` columns and ``NaN`` warm-up
        rows removed.  Column names follow the pattern
        ``'{asset}_ret'``, ``'{asset}_vol'``, ``'{asset}_mom'``.
    """
    assets: list[str] = returns.columns.tolist()

    ret_df = returns.rename(columns={a: f"{a}_ret" for a in assets})
    vol_df = rolling_vol.rename(columns={a: f"{a}_vol" for a in assets})
    mom_df = rolling_mean.rename(columns={a: f"{a}_mom" for a in assets})

    # Stack by type: [all returns | all vols | all moms] — matches paper's
    # x_t = [r_t; sigma_t; m_t] so the first N columns are pure returns.
    ordered_cols: list[str] = (
        [f"{a}_ret" for a in assets]
        + [f"{a}_vol" for a in assets]
        + [f"{a}_mom" for a in assets]
    )

    feature_matrix: pd.DataFrame = (
        pd.concat([ret_df, vol_df, mom_df], axis=1)[ordered_cols].dropna()
    )

    logger.debug(
        "build_feature_matrix: shape=%s, columns=%s",
        feature_matrix.shape,
        feature_matrix.columns.tolist(),
    )
    return feature_matrix


# ---------------------------------------------------------------------------
# Step 4 – causal alignment (main entry point)
# ---------------------------------------------------------------------------


def build_causal_features(
    prices: pd.DataFrame,
    vol_window: int = ROLL_VOL_WINDOW,
    mean_window: int = ROLL_MEAN_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build causally aligned returns and feature matrices from prices.

    Guarantees that ``lagged_features.iloc[t]`` contains only information
    available strictly before day ``t``, making it safe to use as the input
    to a model whose target is the realised return on day ``t``.

    The alignment procedure:

    1. Compute log returns from prices.
    2. Compute rolling vol and rolling mean (unshifted).
    3. Assemble the feature matrix (drops warm-up NaN rows).
    4. Shift the feature matrix forward by one day → ``lagged_features``.
    5. Inner-join all three on their date index and drop any remaining NaN
       rows (removes the single NaN row introduced at the start of
       ``lagged_features`` by the shift).

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed price DataFrame with one column per asset.
    vol_window : int
        Rolling volatility window.  Defaults to :data:`ROLL_VOL_WINDOW`.
    mean_window : int
        Rolling mean window.  Defaults to :data:`ROLL_MEAN_WINDOW`.

    Returns
    -------
    returns : pd.DataFrame
        Log returns aligned to the shared index.  **Not lagged** — used
        as the realised P&L target on each day ``t``.
    features : pd.DataFrame
        Same-day feature matrix (``features.iloc[t]`` uses returns through
        day ``t``).  Useful for post-hoc attribution and research.
    lagged_features : pd.DataFrame
        Features shifted by one day.  ``lagged_features.iloc[t]`` =
        ``features.iloc[t-1]``.  Use this as the model input when
        predicting or trading on day ``t``.

    All three DataFrames share an identical date index.
    """
    returns: pd.DataFrame = compute_log_returns(prices)
    rolling_vol, rolling_mean = compute_rolling_features(
        returns, vol_window, mean_window
    )
    features: pd.DataFrame = build_feature_matrix(returns, rolling_vol, rolling_mean)

    # Shift features forward by one trading day to enforce causality
    lagged_features: pd.DataFrame = features.shift(1)

    # Inner-join on date index so all three cover the same dates, then
    # strip the single leading NaN row produced by the shift
    n_ret: int = returns.shape[1]
    n_feat: int = features.shape[1]

    combined: pd.DataFrame = (
        pd.concat([returns, features, lagged_features], axis=1, join="inner")
        .dropna()
    )

    # Recover individual DataFrames by position (column names may be duplicated
    # between features and lagged_features, so iloc-based slicing is safest)
    returns_aligned: pd.DataFrame = combined.iloc[:, :n_ret].copy()
    features_aligned: pd.DataFrame = combined.iloc[:, n_ret : n_ret + n_feat].copy()
    lagged_aligned: pd.DataFrame = combined.iloc[:, n_ret + n_feat :].copy()

    # Restore original column labels
    returns_aligned.columns = returns.columns
    features_aligned.columns = features.columns
    lagged_aligned.columns = features.columns

    logger.info(
        "build_causal_features: %d rows × %d feature columns "
        "[vol_window=%d, mean_window=%d]",
        len(returns_aligned),
        n_feat,
        vol_window,
        mean_window,
    )

    return returns_aligned, features_aligned, lagged_aligned
