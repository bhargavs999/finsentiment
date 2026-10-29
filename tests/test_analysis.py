"""Aggregation and correlation math."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from finsentiment import analysis


def article(day, score, label="positive", title="t"):
    return {
        "published": datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc),
        "title": title,
        "url": "https://example.com",
        "source": "Test",
        "label": label,
        "score": score,
    }


def test_aggregate_daily_means_and_counts():
    rows = [
        article(date(2026, 10, 1), 0.5),
        article(date(2026, 10, 1), -0.1),
        article(date(2026, 10, 2), 0.8),
    ]
    daily = analysis.aggregate_daily(analysis.articles_to_frame(rows))

    assert list(daily["date"]) == [date(2026, 10, 1), date(2026, 10, 2)]
    assert daily.loc[0, "mean_score"] == pytest.approx(0.2)
    assert list(daily["article_count"]) == [2, 1]


def test_aggregate_daily_drops_rows_without_date_or_score():
    rows = [
        article(date(2026, 10, 1), 0.4),
        {"published": None, "title": "no date", "label": "positive", "score": 0.9},
        {"published": datetime(2026, 10, 2, tzinfo=timezone.utc), "title": "no score", "label": None, "score": None},
    ]
    daily = analysis.aggregate_daily(analysis.articles_to_frame(rows))

    assert len(daily) == 1
    assert daily.loc[0, "mean_score"] == pytest.approx(0.4)


def test_aggregate_daily_handles_empty_input():
    daily = analysis.aggregate_daily(analysis.articles_to_frame([]))
    assert len(daily) == 0
    assert list(daily.columns) == ["date", "mean_score", "article_count"]


def test_join_with_prices_keeps_only_trading_days():
    daily = pd.DataFrame(
        {
            "date": [date(2026, 10, 2), date(2026, 10, 3), date(2026, 10, 5)],  # the 3rd is a Saturday
            "mean_score": [0.1, 0.2, 0.3],
            "article_count": [1, 2, 3],
        }
    )
    prices = pd.DataFrame(
        {
            "date": [date(2026, 10, 2), date(2026, 10, 5)],
            "close": [100.0, 102.0],
            "return_pct": [1.0, 2.0],
            "next_return_pct": [2.0, None],
        }
    )
    merged = analysis.join_with_prices(daily, prices)

    assert list(merged["date"]) == [date(2026, 10, 2), date(2026, 10, 5)]
    assert list(merged["mean_score"]) == [0.1, 0.3]


def test_correlate_perfect_positive_relationship():
    result = analysis.correlate([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])

    assert result["n"] == 5
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_r"] == pytest.approx(1.0)
    assert "Small sample" in result["note"]


def test_correlate_perfect_negative_relationship():
    result = analysis.correlate([1, 2, 3, 4], [10, 7, 4, 1])
    assert result["pearson_r"] == pytest.approx(-1.0)


def test_correlate_monotonic_but_nonlinear_splits_pearson_and_spearman():
    # Spearman sees the perfect ranking; Pearson is dragged down by the curve.
    result = analysis.correlate([1, 2, 3, 4, 5], [1, 2, 4, 8, 64])

    assert result["spearman_r"] == pytest.approx(1.0)
    assert result["pearson_r"] < 0.95


def test_correlate_drops_unpaired_values():
    result = analysis.correlate([1, 2, 3, 4], [1, 2, 3, None])
    assert result["n"] == 3


def test_correlate_refuses_tiny_sample():
    result = analysis.correlate([1, 2], [3, 4])

    assert result["n"] == 2
    assert result["pearson_r"] is None
    assert "at least 3" in result["note"]


def test_correlate_refuses_constant_series():
    result = analysis.correlate([1, 1, 1, 1], [1, 2, 3, 4])

    assert result["pearson_r"] is None
    assert "constant" in result["note"]


def test_correlation_report_covers_both_horizons():
    merged = pd.DataFrame(
        {
            "mean_score": [0.1, 0.2, 0.3, 0.4],
            "return_pct": [0.5, 1.0, 1.5, 2.0],
            "next_return_pct": [1.0, 1.5, 2.0, None],
        }
    )
    rows = analysis.correlation_report(merged)

    assert [r["comparison"] for r in rows] == ["Same-day return", "Next-day return"]
    assert rows[0]["n"] == 4
    assert rows[0]["pearson_r"] == pytest.approx(1.0)
    assert rows[1]["n"] == 3


def test_correlation_report_on_empty_frame_is_safe():
    rows = analysis.correlation_report(pd.DataFrame())
    assert all(r["pearson_r"] is None for r in rows)


def test_label_distribution_and_summary():
    rows = [
        article(date(2026, 10, 1), 0.6, "positive"),
        article(date(2026, 10, 1), -0.7, "negative"),
        article(date(2026, 10, 2), 0.0, "neutral"),
    ]
    frame = analysis.articles_to_frame(rows)

    dist = analysis.label_distribution(frame).set_index("label")["count"].to_dict()
    assert dist == {"positive": 1, "negative": 1, "neutral": 1}

    summary = analysis.summarize(frame)
    assert summary["articles"] == 3
    assert summary["positive"] == 1
    assert summary["negative"] == 1
    assert summary["neutral"] == 1
    assert summary["mean_score"] == pytest.approx(-0.0333, abs=1e-3)
