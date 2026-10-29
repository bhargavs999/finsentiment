"""FinSentiment: score financial news with FinBERT and compare it to price moves.

Educational project. Not financial advice.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finsentiment import analysis, charts, news, prices, sentiment

PRESETS = {
    "AAPL (Apple)": "AAPL",
    "MSFT (Microsoft)": "MSFT",
    "NVDA (NVIDIA)": "NVDA",
    "RELIANCE.NS (Reliance)": "RELIANCE.NS",
    "TCS.NS (TCS)": "TCS.NS",
    "INFY.NS (Infosys)": "INFY.NS",
    "Custom": "",
}

SAMPLE_TEXT = (
    "Revenue for the quarter rose 14 percent year over year, ahead of our guidance. "
    "Operating margin expanded by 180 basis points on better cost control. "
    "We did record a one-time write-down of 45 million on the legacy hardware line. "
    "Net liabilities increased modestly as we drew on the revolving credit facility. "
    "Management is raising full-year guidance and expects demand to stay strong."
)

st.set_page_config(page_title="FinSentiment", page_icon=":chart_with_upwards_trend:", layout="wide")


def _stretch():
    """Streamlit renamed use_container_width to width="stretch" in 1.49."""
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
    except (AttributeError, ValueError):
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 49) else {"use_container_width": True}


STRETCH = _stretch()


# ---------------------------------------------------------------- cached layers

@st.cache_resource(show_spinner="Loading FinBERT (first run downloads about 440 MB)...")
def get_model():
    return sentiment.load_model()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_company_name(ticker: str) -> str:
    return news.resolve_company_name(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_news(ticker: str, company: str, days: int):
    return news.collect_news(ticker, company, days)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(ticker: str, days: int):
    return prices.fetch_prices(ticker, days)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_scores(texts: tuple):
    """Scores for a batch of texts, keyed on the texts so re-runs are free."""
    tokenizer, model = get_model()
    return sentiment.score_texts(list(texts), tokenizer, model)


# ---------------------------------------------------------------------- helpers

def format_corr(value):
    return "n/a" if value is None else "%.3f" % value


def correlation_frame(rows):
    return pd.DataFrame(
        [
            {
                "Comparison": r["comparison"],
                "Pearson r": format_corr(r["pearson_r"]),
                "Pearson p": format_corr(r["pearson_p"]),
                "Spearman rho": format_corr(r["spearman_r"]),
                "Spearman p": format_corr(r["spearman_p"]),
                "n (paired days)": r["n"],
            }
            for r in rows
        ]
    )


def strength_words(r: float) -> str:
    """Plain-language reading of a correlation coefficient."""
    size = abs(r)
    if size < 0.2:
        strength = "essentially no"
    elif size < 0.4:
        strength = "a weak"
    elif size < 0.6:
        strength = "a moderate"
    else:
        strength = "a strong"
    return "%s %s relationship" % (strength, "positive" if r > 0 else "negative")


# ------------------------------------------------------------------- sidebar UI

st.sidebar.title("FinSentiment")
st.sidebar.caption("FinBERT sentiment vs actual price moves")

preset_label = st.sidebar.selectbox("Preset", list(PRESETS.keys()), index=0)
ticker = st.sidebar.text_input(
    "Ticker",
    value=PRESETS[preset_label],
    placeholder="AAPL or RELIANCE.NS",
    help="US symbols as-is (AAPL, MSFT). NSE symbols need the .NS suffix (RELIANCE.NS, TCS.NS).",
).strip().upper()

window_days = st.sidebar.radio(
    "Window", [7, 14, 30], index=0, format_func=lambda d: "%d days" % d, horizontal=True
)
run = st.sidebar.button("Analyze", type="primary", **STRETCH)
st.sidebar.markdown("---")
st.sidebar.caption(
    "News: Google News RSS + yfinance. Prices: yfinance. No API keys needed. "
    "Fetches are cached for 1 hour."
)


# -------------------------------------------------------- tab 1: news vs price

def render_news_tab():
    if not ticker:
        st.info("Pick a preset or type a ticker in the sidebar, then press **Analyze**.")
        return
    if not run and "last_ticker" not in st.session_state:
        st.info("Press **Analyze** in the sidebar to fetch news and prices for %s." % ticker)
        return

    if run:
        st.session_state["last_ticker"] = ticker
        st.session_state["last_window"] = window_days
    active_ticker = st.session_state.get("last_ticker", ticker)
    active_window = st.session_state.get("last_window", window_days)

    company = cached_company_name(active_ticker)
    suffix = " (%s)" % company if company and company != active_ticker else ""
    st.subheader("%s%s - last %d days" % (active_ticker, suffix, active_window))

    # --- news
    with st.spinner("Fetching news..."):
        try:
            articles, note = cached_news(active_ticker, company, active_window)
        except Exception as exc:
            articles, note = [], "news fetch failed (%s)" % exc

    if not articles:
        st.error(
            "No news found for **%s** in the last %d days. Free sources only cover a "
            "recent window, so try a longer window or a more widely covered ticker."
            % (active_ticker, active_window)
        )
        if note:
            st.caption("Source notes: %s" % note)
        return
    if note:
        st.caption("Source notes: %s" % note)

    # --- sentiment
    try:
        with st.spinner("Scoring %d headlines with FinBERT..." % len(articles)):
            scores = cached_scores(tuple(a["title"] for a in articles))
    except Exception as exc:
        st.error("Could not run FinBERT: %s" % exc)
        st.caption("The first run downloads the model, so check the network and retry.")
        return

    scored = [dict(a, **s) for a, s in zip(articles, scores)]
    frame = analysis.articles_to_frame(scored)
    summary = analysis.summarize(frame)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", summary["articles"])
    c2.metric(
        "Mean score",
        "n/a" if summary["mean_score"] is None else "%.3f" % summary["mean_score"],
    )
    c3.metric("Positive", summary["positive"])
    c4.metric("Negative", summary["negative"])

    daily = analysis.aggregate_daily(frame)

    # --- prices
    price_frame = None
    try:
        with st.spinner("Fetching prices..."):
            price_frame = cached_prices(active_ticker, active_window)
    except prices.PriceError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.warning("Price fetch failed: %s" % exc)

    merged = (
        analysis.join_with_prices(daily, price_frame)
        if price_frame is not None
        else pd.DataFrame()
    )

    # --- correlation: the core of the project
    st.markdown("### Sentiment vs return")
    if len(merged) == 0:
        st.info(
            "Not enough overlap between news days and trading days to correlate yet. "
            "The sentiment charts below still work."
        )
    else:
        rows = analysis.correlation_report(merged)
        st.dataframe(correlation_frame(rows), hide_index=True, **STRETCH)

        same_day = rows[0]
        if same_day["pearson_r"] is not None:
            st.markdown(
                "On **%d paired trading days**, daily mean sentiment shows %s with the "
                "same-day return (Pearson r = %.3f, p = %.3f)."
                % (
                    same_day["n"],
                    strength_words(same_day["pearson_r"]),
                    same_day["pearson_r"],
                    same_day["pearson_p"],
                )
            )
        caveats = [r["note"] for r in rows if r["note"]]
        if caveats:
            st.warning(
                "**Read this with care.** %s Free news sources only cover a recent "
                "window, so the sample is short by construction. A correlation over this "
                "few days is not evidence that sentiment predicts price."
                % " ".join(dict.fromkeys(caveats))
            )

    # --- charts
    if len(daily) > 0:
        st.plotly_chart(charts.sentiment_timeline(daily), **STRETCH)
    if len(merged) > 0:
        st.plotly_chart(charts.price_sentiment_overlay(merged), **STRETCH)

    dist = analysis.label_distribution(frame)
    if len(dist) > 0:
        st.plotly_chart(charts.label_distribution_chart(dist), **STRETCH)

    # --- headline table
    st.markdown("### Headlines")
    table = frame[["published", "title", "source", "label", "score", "url"]].copy()
    table["published"] = pd.to_datetime(table["published"], errors="coerce", utc=True)
    table = table.sort_values("score", ascending=False)
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "published": st.column_config.DatetimeColumn("Published", format="YYYY-MM-DD HH:mm"),
            "title": st.column_config.TextColumn("Headline", width="large"),
            "source": st.column_config.TextColumn("Source"),
            "label": st.column_config.TextColumn("Label"),
            "score": st.column_config.NumberColumn("Score", format="%.3f"),
            "url": st.column_config.LinkColumn("Link", display_text="open"),
        },
        **STRETCH,
    )
    st.caption("Click a column header to sort. Score = P(positive) - P(negative).")


# ---------------------------------------------------------- tab 2: free text

def render_text_tab():
    st.subheader("Analyze any financial text")
    st.caption("Paste an earnings-call transcript, a press release or a filing excerpt.")

    text = st.text_area("Text", value="", height=220, placeholder=SAMPLE_TEXT)
    use_sample = st.checkbox("Use the sample text instead")
    if not st.button("Analyze text", type="primary"):
        return

    sentences = sentiment.split_sentences(SAMPLE_TEXT if use_sample else text)
    if not sentences:
        st.warning("Paste at least one full sentence (15 characters or more) to analyze.")
        return

    try:
        with st.spinner("Scoring %d sentences..." % len(sentences)):
            scores = cached_scores(tuple(sentences))
    except Exception as exc:
        st.error("Could not run FinBERT: %s" % exc)
        return

    rows = [dict(s, sentence=sent) for sent, s in zip(sentences, scores)]
    row_frame = pd.DataFrame(rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall score", "%.3f" % float(row_frame["score"].mean()))
    c2.metric("Sentences", len(rows))
    c3.metric("Positive", int((row_frame["label"] == "positive").sum()))
    c4.metric("Negative", int((row_frame["label"] == "negative").sum()))

    dist = row_frame["label"].value_counts().rename_axis("label").reset_index(name="count")
    st.plotly_chart(charts.label_distribution_chart(dist), **STRETCH)
    st.plotly_chart(charts.sentence_score_chart(rows), **STRETCH)

    ranked = row_frame.sort_values("score", ascending=False)
    # Five each, but never the same sentence in both columns: short texts show fewer.
    per_side = max(1, min(5, len(ranked) // 2))

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("#### Most positive")
        for _, r in ranked.head(per_side).iterrows():
            st.markdown("**%+.3f** - %s" % (r["score"], r["sentence"]))
    with col_neg:
        st.markdown("#### Most negative")
        for _, r in ranked.tail(per_side).iloc[::-1].iterrows():
            st.markdown("**%+.3f** - %s" % (r["score"], r["sentence"]))
    if per_side < 5:
        st.caption("Showing %d each, since the text has only %d scored sentences." % (per_side, len(ranked)))

    with st.expander("All sentences"):
        st.dataframe(
            row_frame[["sentence", "label", "score", "p_positive", "p_negative", "p_neutral"]],
            hide_index=True,
            **STRETCH,
        )


tab_news, tab_text = st.tabs(["News & Price Correlation", "Analyze Text"])
with tab_news:
    render_news_tab()
with tab_text:
    render_text_tab()

st.markdown("---")
st.caption("Educational project. Not financial advice.")
