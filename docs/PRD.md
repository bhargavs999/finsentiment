# FinSentiment - Product Requirements

## Problem

Financial news moves prices, but a student or retail investor reading it manually
cannot keep up, and general-purpose sentiment models misread finance vocabulary.
"Liability", "write-down", "impairment" and "beat estimates" carry a specific
meaning in a financial context that a model trained on movie reviews gets wrong.
FinBERT is trained on financial text, so it reads them correctly.

The second question this project asks is whether that sentiment lines up with what
the price actually did. That is the part most dashboards skip.

## Requirements

### Functional

| # | Requirement |
|---|-------------|
| F1 | Ticker input supporting US symbols (`AAPL`) and NSE symbols (`RELIANCE.NS`), with presets and a 7 / 14 / 30-day window |
| F2 | Fetch news with no API keys: Google News RSS plus `yfinance.Ticker.news`, deduped by normalized title |
| F3 | Score each article with FinBERT (`ProsusAI/finbert`) on CPU, batched, truncated at 512 tokens; emit label, class probabilities and `score = P(pos) - P(neg)` |
| F4 | Aggregate sentiment per day (mean score + article count); fetch prices and compute daily returns |
| F5 | Report Pearson and Spearman correlation of daily sentiment against same-day and next-day returns, always with sample size `n` and a caveat when `n` is small |
| F6 | Charts: sentiment timeline, price-vs-sentiment overlay, label distribution, sortable headline table with scores and links |
| F7 | Second tab: score pasted text sentence by sentence, with overall score, label breakdown, and top five positive / negative sentences |
| F8 | Footer disclaimer: "Educational project. Not financial advice." |

### Non-functional

- No API keys, no account, no database. Everything runs from a clone and a pip install.
- Model loads once per session (`st.cache_resource`); network fetches cached for one
  hour (`st.cache_data(ttl=3600)`).
- Errors surface as readable messages in the UI, never as a raw traceback.
- CPU only; no GPU assumed.
- Pinned dependency versions.

### Out of scope

Authentication, a database, model training or fine-tuning, a model-evaluation tab,
and anything real-time.

## Architecture

```text
app.py                  Streamlit UI: sidebar, two tabs, caching layer
finsentiment/
  news.py               Google News RSS + yfinance news, company-name resolution,
                        title normalization, dedupe, window filtering
  sentiment.py          FinBERT load/score (batched, 512-token truncation),
                        sentence splitting
  prices.py             yfinance daily closes, daily and next-day returns
  analysis.py           daily aggregation, join onto trading days, Pearson/Spearman,
                        label distribution, summary metrics
  charts.py             Plotly figures
tests/                  pytest suite; the model is mocked, weights never download
```

Data flow for tab 1:

```text
ticker -> company name (yfinance)
       -> Google News RSS + yfinance news -> dedupe -> window filter
       -> FinBERT batch scoring -> per-article label + score
       -> aggregate per day (mean score, count)
       -> inner join onto trading days (yfinance closes + returns)
       -> Pearson / Spearman vs same-day and next-day return
       -> charts + headline table
```

The `finsentiment/` package holds no Streamlit calls except the model loader's
callers, so the aggregation and correlation logic is testable without a UI.

## Data sources

| Source | Used for | Keys | Notes |
|---|---|---|---|
| Google News RSS (`news.google.com/rss/search`) | Headlines | None | Query is `("<company>" OR <TICKER>) stock when:<N>d`; parsed with feedparser |
| `yfinance.Ticker.news` | Headlines | None | Item shape changed in yfinance 0.2.54; both the old flat shape and the new nested `content` shape are handled |
| `yfinance.Ticker.info` | Company name for the news query | None | Falls back to the bare ticker when the lookup fails |
| `yfinance.download` | Daily closes and returns | None | `auto_adjust=True`; start padded by 7 days so the first in-window day has a prior close |

Fallback behaviour: if either news source returns nothing the app continues with
the other and names the dead source in the UI ("Source notes"). If both return
nothing, the UI explains that free sources only cover a recent window. If the price
fetch fails, sentiment charts still render and the correlation section says why it
is empty.

## Limitations

1. **Sample size.** Free news covers a recent window only, so a 7-day run usually
   yields single-digit paired trading days. Correlations are reported with `n` and
   flagged below `n = 10`; below `n = 3` they are not computed at all.
2. **Headlines, not bodies.** Both feeds supply titles, so the model never sees the
   article text.
3. **Day-level alignment.** Articles are bucketed by UTC calendar day and joined to
   trading days. Whether a story landed before or after the close is ignored, which
   matters most for NSE tickers because IST leads UTC by 5h30m.
4. **Weekend news is dropped from the correlation** by the inner join, since there
   is no same-day return to pair it with.
5. **Headline-level FinBERT is a blunt instrument.** It was trained on financial
   phrases, not clickbait headlines, and sarcasm or forward-looking language can
   flip a label.
6. **No causal claim.** Correlation here is descriptive, and nothing is backtested.

## Decisions log

| Decision | Reasoning |
|---|---|
| Fork `eyllasmin/fin-sentiment-tracker` (MIT) rather than build fresh | It is the closest permissively-licensed FinBERT + Streamlit + Plotly project, and its compound score `P(pos) - P(neg)` is exactly the scoring this project needs. Lineage stays visible as a GitHub fork. |
| Drop upstream's SQLite layer | The brief rules out a database; Streamlit's `cache_data` covers the one-hour reuse this app needs. |
| Replace upstream's Finviz scraper | Finviz HTML scraping is brittle and blocks automated clients; Google News RSS plus yfinance need no keys and no scraping. |
| Score = `P(positive) - P(negative)` | One number spanning -1 to +1, and neutral articles sit near zero without needing a separate rule. |
| Read label order from `model.config.id2label` | Hard-coding the order breaks silently if the model revision reorders classes; the config is authoritative, with `["positive", "negative", "neutral"]` as a fallback. |
| Inner join sentiment onto trading days | An outer join would pad the series with weekend days that have no return, inflating `n` on a sample that is already small. |
| Refuse to compute correlation below `n = 3`, warn below `n = 10` | Pearson on two points is always +/-1 and would be read as a real result. |
| Both Pearson and Spearman | Pearson catches a linear relationship, Spearman catches a monotonic one and is far less sensitive to the single large return that a short window often contains. |
| Dual-axis price-vs-sentiment chart | Two y-scales is normally poor practice (and the in-house dataviz guidance rules it out), but the brief asks for the overlay, and price and sentiment share no unit. Mitigated with per-series axis-title colours, a dashed sentiment line and a unified hover. |
| Blue / red / gray for positive / negative / neutral | The finance convention is green/red, which is the worst possible pairing for red-green colour blindness. Blue-red with a gray neutral is a diverging scale that survives colour-vision deficiency, and it passed the palette validator. |
| Dedupe by normalized title, not URL | Google News and Yahoo link to the same story through different URLs; the title is the stable key after lowercasing and stripping punctuation. |
| `st.cache_data` keyed on the tuple of texts for scoring | Switching tabs or re-sorting re-runs the script; keying on the texts keeps FinBERT from re-scoring the same headlines. |
| Python 3.11 via `uv` | No system pip or venv module was present on this machine; `uv` provided both the interpreter and the installer. |
