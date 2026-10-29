"""Daily aggregation and the sentiment-vs-return correlation."""

from __future__ import annotations

import pandas as pd
from scipy import stats

MIN_N_FOR_CORRELATION = 3
SMALL_SAMPLE_N = 10

ARTICLE_COLUMNS = ["published", "title", "url", "source", "label", "score"]


def articles_to_frame(articles) -> pd.DataFrame:
    """Scored articles to a DataFrame with a plain `date` column (UTC calendar day)."""
    frame = pd.DataFrame(list(articles))
    for col in ARTICLE_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    if len(frame) == 0:
        frame["date"] = pd.Series(dtype="object")
        return frame

    published = pd.to_datetime(frame["published"], errors="coerce", utc=True)
    frame["date"] = published.dt.date
    return frame


def aggregate_daily(articles_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean sentiment score and article count per day, oldest first.

    Articles with no date or no score cannot be lined up against a trading day,
    so they are dropped here (they still show in the headline table).
    """
    if len(articles_frame) == 0 or "date" not in articles_frame.columns:
        return pd.DataFrame(columns=["date", "mean_score", "article_count"])

    frame = articles_frame.dropna(subset=["date", "score"]).copy()
    if len(frame) == 0:
        return pd.DataFrame(columns=["date", "mean_score", "article_count"])

    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["score"])

    daily = (
        frame.groupby("date", as_index=False)
        .agg(mean_score=("score", "mean"), article_count=("score", "size"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def join_with_prices(daily: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Inner-join daily sentiment onto trading days.

    An inner join is deliberate: news published on a weekend has no same-day return,
    and inventing one would inflate the sample the correlation is computed over.
    """
    if len(daily) == 0 or len(prices) == 0:
        return pd.DataFrame(
            columns=["date", "mean_score", "article_count", "close", "return_pct", "next_return_pct"]
        )

    left = daily.copy()
    right = prices.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.date
    right["date"] = pd.to_datetime(right["date"]).dt.date

    merged = left.merge(right, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return merged


def correlate(x, y) -> dict:
    """Pearson and Spearman correlation of two aligned series.

    Returns n plus coefficients and p-values, or None values when the sample is too
    small or one series is constant (correlation is undefined in both cases).
    """
    frame = pd.DataFrame({"x": pd.to_numeric(pd.Series(list(x)), errors="coerce"),
                          "y": pd.to_numeric(pd.Series(list(y)), errors="coerce")}).dropna()
    n = len(frame)
    result = {
        "n": n,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_r": None,
        "spearman_p": None,
        "note": "",
    }

    if n < MIN_N_FOR_CORRELATION:
        result["note"] = "Need at least %d paired days; have %d." % (MIN_N_FOR_CORRELATION, n)
        return result
    if frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        result["note"] = "One series is constant, so correlation is undefined."
        return result

    pearson = stats.pearsonr(frame["x"], frame["y"])
    spearman = stats.spearmanr(frame["x"], frame["y"])
    result["pearson_r"] = float(pearson[0])
    result["pearson_p"] = float(pearson[1])
    result["spearman_r"] = float(spearman[0])
    result["spearman_p"] = float(spearman[1])
    if n < SMALL_SAMPLE_N:
        result["note"] = "Small sample (n=%d): treat this as indicative only." % n
    return result


def correlation_report(merged: pd.DataFrame) -> list[dict]:
    """Correlations of daily sentiment against same-day and next-day returns."""
    rows = []
    for key, label in (("return_pct", "Same-day return"), ("next_return_pct", "Next-day return")):
        series_y = merged[key] if key in merged.columns else []
        stats_row = correlate(merged.get("mean_score", []), series_y)
        stats_row["comparison"] = label
        rows.append(stats_row)
    return rows


def label_distribution(articles_frame: pd.DataFrame) -> pd.DataFrame:
    """Article count per FinBERT label."""
    if len(articles_frame) == 0 or "label" not in articles_frame.columns:
        return pd.DataFrame(columns=["label", "count"])
    counts = (
        articles_frame["label"].dropna().value_counts().rename_axis("label").reset_index(name="count")
    )
    return counts


def summarize(articles_frame: pd.DataFrame) -> dict:
    """Headline numbers for the metric row."""
    scores = pd.to_numeric(articles_frame.get("score", pd.Series(dtype=float)), errors="coerce").dropna()
    labels = articles_frame.get("label", pd.Series(dtype=object)).dropna()
    total = len(articles_frame)
    positive = int((labels == "positive").sum())
    negative = int((labels == "negative").sum())
    return {
        "articles": total,
        "mean_score": float(scores.mean()) if len(scores) else None,
        "positive": positive,
        "negative": negative,
        "neutral": int(len(labels)) - positive - negative,
    }
