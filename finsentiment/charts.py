"""Plotly figures for the dashboard.

Palette: sentiment is polarity, so it uses a diverging blue/red pair with a gray
neutral (colorblind-safe, unlike the usual green/red). Price uses orange so it
never collides with the sentiment blue.
"""

from __future__ import annotations

import plotly.graph_objects as go

POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
NEUTRAL = "#898781"
PRICE = "#eb6834"

GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED_INK = "#898781"

LABEL_COLORS = {"positive": POSITIVE, "negative": NEGATIVE, "neutral": NEUTRAL}


def _base_layout(fig, height=360, title=None):
    fig.update_layout(
        height=height,
        title=title,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED_INK))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED_INK))
    return fig


def sentiment_timeline(daily):
    """Mean daily sentiment over time, with a zero line to read polarity against."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(daily["date"]),
            y=list(daily["mean_score"]),
            mode="lines+markers",
            name="Mean sentiment",
            line=dict(color=POSITIVE, width=2),
            marker=dict(size=8, color=POSITIVE, line=dict(width=2, color="#fcfcfb")),
            customdata=list(daily["article_count"]),
            hovertemplate="%{x}<br>Mean score %{y:.3f}<br>%{customdata} article(s)<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    _base_layout(fig, title="Daily mean sentiment")
    fig.update_yaxes(title="Score (-1 to +1)", range=[-1.05, 1.05])
    return fig


def price_sentiment_overlay(merged):
    """Close price and mean sentiment on a shared date axis.

    Two y-scales on one plot is normally the wrong call, but the two measures have
    no common unit and reading them day-against-day is the point of the project.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(merged["date"]),
            y=list(merged["close"]),
            mode="lines+markers",
            name="Close price",
            line=dict(color=PRICE, width=2),
            marker=dict(size=7, color=PRICE, line=dict(width=2, color="#fcfcfb")),
            hovertemplate="%{x}<br>Close %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(merged["date"]),
            y=list(merged["mean_score"]),
            mode="lines+markers",
            name="Mean sentiment",
            yaxis="y2",
            line=dict(color=POSITIVE, width=2, dash="dot"),
            marker=dict(size=7, color=POSITIVE, line=dict(width=2, color="#fcfcfb")),
            hovertemplate="%{x}<br>Sentiment %{y:.3f}<extra></extra>",
        )
    )
    _base_layout(fig, title="Price vs sentiment")
    fig.update_yaxes(title=dict(text="Close price", font=dict(color=PRICE)))
    fig.update_layout(
        yaxis2=dict(
            title=dict(text="Mean sentiment", font=dict(color=POSITIVE)),
            overlaying="y",
            side="right",
            range=[-1.05, 1.05],
            showgrid=False,
            tickfont=dict(color=MUTED_INK),
        )
    )
    return fig


def label_distribution_chart(dist):
    """Article count per FinBERT label, counts written at the data-ends."""
    labels = [str(l) for l in dist["label"]]
    counts = [int(c) for c in dist["count"]]
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker=dict(color=[LABEL_COLORS.get(l, NEUTRAL) for l in labels]),
            text=counts,
            textposition="outside",
            hovertemplate="%{y}: %{x} article(s)<extra></extra>",
        )
    )
    _base_layout(fig, height=260, title="Label distribution")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, title="Articles")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def sentence_score_chart(rows):
    """Sentence-level scores for the Analyze Text tab, in document order."""
    fig = go.Figure(
        go.Bar(
            x=list(range(1, len(rows) + 1)),
            y=[r["score"] for r in rows],
            marker=dict(color=[LABEL_COLORS.get(r["label"], NEUTRAL) for r in rows]),
            customdata=[[r["label"], r["sentence"][:140]] for r in rows],
            hovertemplate="Sentence %{x} (%{customdata[0]})<br>Score %{y:.3f}<br>%{customdata[1]}<extra></extra>",
        )
    )
    _base_layout(fig, height=300, title="Sentence-by-sentence score")
    fig.update_xaxes(title="Sentence number")
    fig.update_yaxes(title="Score (-1 to +1)", range=[-1.05, 1.05])
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig
