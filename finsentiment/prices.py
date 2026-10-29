"""Daily prices and returns via yfinance."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf


class PriceError(RuntimeError):
    """Raised when prices for a ticker cannot be fetched or are empty."""


def fetch_prices(ticker: str, days: int, end: date | None = None) -> pd.DataFrame:
    """Daily closes for the window.

    Pads the start by a week so the first in-window day still has a previous close
    to compute a return against, and so weekends/holidays do not empty the frame.
    """
    end = end or date.today()
    start = end - timedelta(days=days + 7)
    try:
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise PriceError("Could not fetch prices for %s: %s" % (ticker, exc)) from exc

    if raw is None or len(raw) == 0:
        raise PriceError(
            "No price data for '%s'. Check the ticker: US symbols look like AAPL, "
            "NSE symbols need the .NS suffix (RELIANCE.NS)." % ticker
        )

    # yfinance returns MultiIndex columns when more than one field/ticker comes back.
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(-1, axis=1)

    frame = raw.reset_index()
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]

    # yfinance returns the index tz-aware for some exchanges and naive for others;
    # tz_localize(None) raises on an already-naive series, so only drop a real tz.
    dates = pd.to_datetime(frame[date_col])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "date": dates.dt.date,
            "close": pd.to_numeric(frame["Close"], errors="coerce"),
        }
    ).dropna()
    return add_returns(out)


def add_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add daily return (close-to-close pct change) and the next day's return."""
    out = prices.sort_values("date").reset_index(drop=True).copy()
    out["return_pct"] = out["close"].pct_change() * 100.0
    out["next_return_pct"] = out["return_pct"].shift(-1)
    return out
