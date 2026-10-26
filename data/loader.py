"""data/loader.py — Yahoo Finance price loader with parquet caching.

Provides a canonical TICKER_MANIFEST and two public functions:
  - download_prices: fetches adjusted close prices via yfinance
  - load_prices: wraps download_prices with a local parquet cache
"""

import logging
import os

import pandas as pd
import yfinance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------

TICKER_MANIFEST: dict[str, str] = {
    "SPX": "SPY",
    "BOND": "TLT",
    "GOLD": "GLD",
    "OIL": "USO",
    "USD": "UUP",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_prices(
    start: str = "2005-01-01",
    end: str = "2026-01-31",
    tickers: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Fetches daily adjusted close prices for every ticker in *tickers* over
    the half-open interval [start, end).  Rows where **any** asset is missing
    a price are dropped so that the returned DataFrame is always complete.

    Parameters
    ----------
    start : str
        Inclusive start date in ``'YYYY-MM-DD'`` format.
    end : str
        Exclusive end date in ``'YYYY-MM-DD'`` format.
    tickers : dict[str, str] | None
        Mapping of human-readable asset name → Yahoo Finance ticker symbol.
        Defaults to :data:`TICKER_MANIFEST` when *None*.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame of adjusted close prices.  Column labels are
        the **asset names** (keys of *tickers*), ordered to match the dict.
        All rows with at least one missing value are dropped.

    Raises
    ------
    ValueError
        If yfinance returns no data for the requested date range.
    """
    if tickers is None:
        tickers = TICKER_MANIFEST

    asset_names: list[str] = list(tickers.keys())
    ticker_symbols: list[str] = list(tickers.values())

    logger.info(
        "Downloading prices for %s [%s → %s]",
        ticker_symbols,
        start,
        end,
    )

    raw: pd.DataFrame = yfinance.download(
        ticker_symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for tickers={ticker_symbols}, "
            f"start={start!r}, end={end!r}."
        )

    # yfinance ≥1.4 with multi_level_index=True (default) returns a MultiIndex
    # DataFrame whose top level is the price field ('Close', 'Open', …) and
    # whose second level is the ticker symbol.
    if isinstance(raw.columns, pd.MultiIndex):
        prices: pd.DataFrame = raw["Close"].copy()
    else:
        # Single-ticker edge-case: plain column named 'Close'
        prices = raw[["Close"]].copy()

    # Rename ticker symbols → asset names for readability downstream
    symbol_to_name: dict[str, str] = {sym: name for name, sym in tickers.items()}
    prices = prices.rename(columns=symbol_to_name)

    # Enforce manifest key order
    prices = prices[asset_names]

    logger.info("Downloaded %d rows before dropping missing data.", len(prices))

    missing_counts = prices.isna().sum()
    if missing_counts.any():
        logger.warning(
            "Missing values per asset before row-drop:\n%s",
            missing_counts[missing_counts > 0].to_string(),
        )

    prices = prices.dropna()
    logger.info("Retained %d rows after dropping incomplete rows.", len(prices))

    return prices


def load_prices(
    cache_path: str = "data/prices.parquet",
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load adjusted close prices, using a local parquet cache when available.

    On first call (or when *force_reload* is ``True``) the function delegates
    to :func:`download_prices`, persists the result to *cache_path*, and
    returns it.  On subsequent calls it reads directly from the parquet file,
    avoiding unnecessary network round-trips.

    Parameters
    ----------
    cache_path : str
        Filesystem path for the parquet cache file.
        Defaults to ``'data/prices.parquet'``.
    force_reload : bool
        When ``True``, ignore the existing cache and re-download from Yahoo
        Finance.  Defaults to ``False``.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame of adjusted close prices (see
        :func:`download_prices` for column layout).
    """
    if not force_reload and os.path.exists(cache_path):
        logger.info("Loading prices from parquet cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    prices = download_prices()

    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    prices.to_parquet(cache_path)
    logger.info("Prices cached to: %s", cache_path)

    return prices
