"""News fetching with no API keys: Google News RSS + yfinance's own news feed."""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser
import yfinance as yf

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# Tickers get an exchange suffix on Yahoo (RELIANCE.NS); strip it for the news query.
_SUFFIX_RE = re.compile(r"\.(NS|BO|L|TO|AX|HK|DE|PA)$", re.IGNORECASE)

# Suffixes yfinance returns in company names that add nothing to a news search.
_NAME_NOISE_RE = re.compile(
    r"\b(limited|ltd|inc|incorporated|corp|corporation|company|co|plc|sa|nv|ag|holdings?)\b\.?",
    re.IGNORECASE,
)

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def base_ticker(ticker: str) -> str:
    """AAPL -> AAPL, RELIANCE.NS -> RELIANCE."""
    return _SUFFIX_RE.sub("", ticker.strip().upper())


def resolve_company_name(ticker: str) -> str:
    """Company name from yfinance, falling back to the bare ticker when lookup fails."""
    fallback = base_ticker(ticker)
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return fallback
    name = info.get("shortName") or info.get("longName") or ""
    name = _NAME_NOISE_RE.sub("", str(name)).strip(" .,-")
    name = _WS_RE.sub(" ", name)
    return name or fallback


def build_query(ticker: str, company: str, days: int) -> str:
    """Google News query: company name plus ticker, limited to the window."""
    bt = base_ticker(ticker)
    parts = []
    if company and company.upper() != bt:
        parts.append('"%s"' % company)
    parts.append(bt)
    return "(%s) stock when:%dd" % (" OR ".join(parts), days)


def normalize_title(title: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace, for dedupe keys."""
    t = (title or "").lower()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def dedupe_articles(articles):
    """Keep the first article for each normalized title; drop empty titles."""
    seen = set()
    out = []
    for a in articles:
        key = normalize_title(a.get("title", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def _as_utc(value) -> datetime | None:
    """Best-effort conversion of the many timestamp shapes these feeds return."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return None
    if isinstance(value, time.struct_time):
        return datetime.fromtimestamp(time.mktime(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            return _as_utc(datetime.fromisoformat(text))
        except ValueError:
            pass
        try:
            from email.utils import parsedate_to_datetime

            return _as_utc(parsedate_to_datetime(value))
        except (TypeError, ValueError):
            return None
    return None


def fetch_google_news(ticker: str, company: str, days: int = 7):
    """Articles from the Google News RSS search feed. Returns [] on any failure."""
    query = build_query(ticker, company, days)
    url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []

    articles = []
    for entry in getattr(feed, "entries", []):
        published = _as_utc(
            getattr(entry, "published_parsed", None) or entry.get("published")
        )
        source = ""
        src = entry.get("source")
        if isinstance(src, dict):
            source = src.get("title", "")
        articles.append(
            {
                "title": (entry.get("title") or "").strip(),
                "url": entry.get("link", ""),
                "source": source or "Google News",
                "published": published,
            }
        )
    return articles


def fetch_yfinance_news(ticker: str):
    """Articles from yfinance's news feed, handling both old and new item shapes."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    articles = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # yfinance >=0.2.54 nests the article under "content".
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        title = content.get("title") or item.get("title") or ""
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else None
        ) or (
            (content.get("clickThroughUrl") or {}).get("url")
            if isinstance(content.get("clickThroughUrl"), dict)
            else None
        ) or item.get("link", "")
        provider = content.get("provider")
        source = (
            provider.get("displayName")
            if isinstance(provider, dict)
            else item.get("publisher", "")
        )
        published = _as_utc(
            content.get("pubDate")
            or content.get("displayTime")
            or item.get("providerPublishTime")
        )
        articles.append(
            {
                "title": str(title).strip(),
                "url": url,
                "source": source or "Yahoo Finance",
                "published": published,
            }
        )
    return articles


def within_window(articles, days: int, now: datetime | None = None):
    """Drop articles older than the window. Undated articles are kept."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    return [a for a in articles if a.get("published") is None or a["published"] >= cutoff]


def collect_news(ticker: str, company: str, days: int = 7):
    """Both sources, deduped by title, newest first. Second return value is a note."""
    google = fetch_google_news(ticker, company, days)
    yahoo = fetch_yfinance_news(ticker)

    notes = []
    if not google:
        notes.append("Google News RSS returned nothing")
    if not yahoo:
        notes.append("yfinance news returned nothing")

    merged = dedupe_articles(google + yahoo)
    merged = within_window(merged, days)
    merged.sort(key=lambda a: a.get("published") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return merged, "; ".join(notes)
