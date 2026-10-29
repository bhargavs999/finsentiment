"""Query building, title normalization, dedupe and timestamp parsing."""

from datetime import datetime, timedelta, timezone

from finsentiment import news


def test_base_ticker_strips_exchange_suffix():
    assert news.base_ticker("AAPL") == "AAPL"
    assert news.base_ticker("reliance.ns") == "RELIANCE"
    assert news.base_ticker(" tcs.NS ") == "TCS"


def test_build_query_includes_company_and_ticker_and_window():
    query = news.build_query("RELIANCE.NS", "Reliance Industries", 14)

    assert '"Reliance Industries"' in query
    assert "RELIANCE" in query
    assert "when:14d" in query


def test_build_query_without_company_name():
    query = news.build_query("AAPL", "AAPL", 7)

    assert query.count("AAPL") == 1
    assert "when:7d" in query


def test_normalize_title_ignores_case_and_punctuation():
    a = news.normalize_title("Apple beats estimates -- again!")
    b = news.normalize_title("apple beats estimates   again")
    assert a == b == "apple beats estimates again"


def test_dedupe_keeps_first_and_drops_blank_titles():
    articles = [
        {"title": "Infosys wins deal", "source": "Google News"},
        {"title": "INFOSYS WINS DEAL!", "source": "Yahoo Finance"},
        {"title": "", "source": "Yahoo Finance"},
        {"title": "Infosys cuts guidance", "source": "Google News"},
    ]
    out = news.dedupe_articles(articles)

    assert [a["title"] for a in out] == ["Infosys wins deal", "Infosys cuts guidance"]
    assert out[0]["source"] == "Google News"


def test_within_window_filters_old_articles_but_keeps_undated():
    now = datetime(2026, 10, 29, tzinfo=timezone.utc)
    articles = [
        {"title": "fresh", "published": now - timedelta(days=2)},
        {"title": "stale", "published": now - timedelta(days=30)},
        {"title": "undated", "published": None},
    ]
    kept = [a["title"] for a in news.within_window(articles, days=7, now=now)]

    assert kept == ["fresh", "undated"]


def test_as_utc_parses_the_shapes_these_feeds_return():
    expected = datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc)

    assert news._as_utc("2026-10-20T13:30:00Z") == expected
    assert news._as_utc("Tue, 20 Oct 2026 13:30:00 GMT") == expected
    assert news._as_utc(expected.timestamp()) == expected
    assert news._as_utc(datetime(2026, 10, 20, 13, 30)) == expected
    assert news._as_utc(None) is None
    assert news._as_utc("not a date") is None


def test_fetch_google_news_maps_entries(monkeypatch):
    class FakeFeed:
        entries = [
            {
                "title": "  TCS beats estimates  ",
                "link": "https://example.com/a",
                "published": "Tue, 20 Oct 2026 13:30:00 GMT",
                "source": {"title": "Reuters"},
            }
        ]

    monkeypatch.setattr(news.feedparser, "parse", lambda url: FakeFeed())
    out = news.fetch_google_news("TCS.NS", "Tata Consultancy Services", 7)

    assert len(out) == 1
    assert out[0]["title"] == "TCS beats estimates"
    assert out[0]["source"] == "Reuters"
    assert out[0]["published"] == datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc)


def test_fetch_google_news_returns_empty_when_feed_breaks(monkeypatch):
    def boom(url):
        raise OSError("network down")

    monkeypatch.setattr(news.feedparser, "parse", boom)
    assert news.fetch_google_news("AAPL", "Apple", 7) == []


def test_fetch_yfinance_news_handles_new_nested_shape(monkeypatch):
    items = [
        {
            "content": {
                "title": "Apple ships record quarter",
                "pubDate": "2026-10-21T09:00:00Z",
                "canonicalUrl": {"url": "https://example.com/apple"},
                "provider": {"displayName": "Bloomberg"},
            }
        }
    ]
    monkeypatch.setattr(news.yf, "Ticker", lambda t: type("T", (), {"news": items})())
    out = news.fetch_yfinance_news("AAPL")

    assert out[0]["title"] == "Apple ships record quarter"
    assert out[0]["url"] == "https://example.com/apple"
    assert out[0]["source"] == "Bloomberg"
    assert out[0]["published"] == datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc)


def test_fetch_yfinance_news_handles_legacy_flat_shape(monkeypatch):
    items = [
        {
            "title": "Legacy item",
            "link": "https://example.com/legacy",
            "publisher": "Yahoo",
            "providerPublishTime": datetime(2026, 10, 21, 9, 0, tzinfo=timezone.utc).timestamp(),
        }
    ]
    monkeypatch.setattr(news.yf, "Ticker", lambda t: type("T", (), {"news": items})())
    out = news.fetch_yfinance_news("AAPL")

    assert out[0]["title"] == "Legacy item"
    assert out[0]["url"] == "https://example.com/legacy"
    assert out[0]["source"] == "Yahoo"


def test_collect_news_merges_dedupes_and_notes_dead_sources(monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        news,
        "fetch_google_news",
        lambda t, c, d: [
            {"title": "Shared headline", "url": "g", "source": "Google News", "published": now},
            {"title": "Google only", "url": "g2", "source": "Google News", "published": now - timedelta(days=1)},
        ],
    )
    monkeypatch.setattr(news, "fetch_yfinance_news", lambda t: [])

    merged, note = news.collect_news("AAPL", "Apple", 7)

    assert [a["title"] for a in merged] == ["Shared headline", "Google only"]
    assert "yfinance" in note


def test_resolve_company_name_falls_back_when_lookup_fails(monkeypatch):
    def boom(t):
        raise OSError("no network")

    monkeypatch.setattr(news.yf, "Ticker", boom)
    assert news.resolve_company_name("RELIANCE.NS") == "RELIANCE"


def test_resolve_company_name_trims_corporate_suffixes(monkeypatch):
    monkeypatch.setattr(
        news.yf,
        "Ticker",
        lambda t: type("T", (), {"info": {"shortName": "Reliance Industries Limited"}})(),
    )
    assert news.resolve_company_name("RELIANCE.NS") == "Reliance Industries"
