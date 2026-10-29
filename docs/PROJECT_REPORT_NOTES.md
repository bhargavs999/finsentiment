# FinSentiment - Project Report Notes

Notes for the written submission. Paste and edit as needed.

## Title

FinSentiment: Domain-Specific Sentiment Analysis of Financial News with FinBERT
and Its Relationship to Stock Price Movement

## Abstract

Financial news influences stock prices, but the volume of coverage makes manual
reading impractical, and general-purpose sentiment models misinterpret financial
vocabulary: terms such as "liability", "write-down" and "beat estimates" carry a
domain-specific meaning that a model trained on consumer reviews does not capture.
This project implements FinSentiment, a Streamlit web application that collects
recent news for a chosen stock from two keyless public sources (Google News RSS
and the yfinance news feed), scores each headline with FinBERT (`ProsusAI/finbert`),
a BERT model fine-tuned on financial text, and aggregates those scores into a daily
sentiment series. The application then fetches daily closing prices for the same
ticker through yfinance, computes daily returns, and measures the Pearson and
Spearman correlation between daily mean sentiment and both the same-day and the
next-day return. Results are presented with the sample size and an explicit
small-sample caveat, since freely available news covers only a recent window. A
second module scores arbitrary financial text sentence by sentence, reporting an
overall score and the most positive and most negative sentences. The system
supports both US and NSE-listed tickers, requires no API keys, database or model
training, and runs entirely on CPU. It is intended as an educational tool for
understanding domain-specific NLP, not as financial advice.

## Problem statement

General-purpose sentiment analysis fails on financial text because the same words
carry different polarity in a financial context. A retail investor or student who
wants to understand the tone of recent coverage for a stock faces two problems:
the volume of news is too large to read manually, and off-the-shelf sentiment
tools misclassify finance-specific language. Further, most existing sentiment
dashboards stop at displaying sentiment and never ask the obvious follow-up
question - whether the measured sentiment corresponds to the price movement that
actually occurred.

## Objectives

1. Collect recent news for a given stock ticker from public sources that require
   no API key, registration or paid plan.
2. Support both US-listed (`AAPL`, `MSFT`) and NSE-listed (`RELIANCE.NS`,
   `TCS.NS`, `INFY.NS`) securities.
3. Classify each article with FinBERT, a transformer fine-tuned on financial text,
   producing a label, class probabilities and a single continuous score.
4. Aggregate article-level sentiment into a daily series and align it with daily
   stock returns obtained from the same free data source.
5. Quantify the relationship between sentiment and return using both a parametric
   (Pearson) and a rank-based (Spearman) correlation measure, at same-day and
   next-day horizons, and report it honestly with its sample size.
6. Visualize the result through an interactive dashboard.
7. Provide a general-purpose text analysis mode for transcripts and press releases.

## Methodology

**1. News acquisition.** The company name is resolved from the ticker through
`yfinance.Ticker.info` and corporate suffixes ("Limited", "Inc") are stripped. A
Google News RSS query of the form `("<company>" OR <TICKER>) stock when:<N>d` is
parsed with feedparser, and the `yfinance.Ticker.news` feed is read for the same
ticker. Both feeds are merged and deduplicated on a normalized title key
(lowercased, punctuation removed, whitespace collapsed), because the two sources
frequently carry the same story under different URLs. Articles older than the
selected window are discarded.

**2. Sentiment classification.** Headlines are scored with `ProsusAI/finbert`
through the Hugging Face Transformers library on CPU. Texts are tokenized in
batches with padding and truncation at 512 tokens, the model's logits are converted
to probabilities with softmax, and each article receives:

- a label, the arg-max class (positive / negative / neutral),
- the three class probabilities, and
- a compound score, `score = P(positive) - P(negative)`, which maps the
  three-class output onto a single continuous scale from -1 to +1 and places
  neutral articles near zero.

The label order is read from `model.config.id2label` rather than hard-coded, so a
reordered model revision cannot silently invert the results.

**3. Aggregation.** Articles are bucketed by UTC calendar day; each day yields a
mean score and an article count.

**4. Price data and returns.** Daily closing prices are fetched with
`yfinance.download` using `auto_adjust=True`, with the window start padded by seven
days so the first in-window day still has a prior close. The daily return is the
close-to-close percentage change, and the next-day return is that series shifted
by one position.

**5. Alignment and correlation.** The daily sentiment series is inner-joined onto
trading days, so only days with both news and a traded return enter the analysis;
this avoids inflating the sample with weekend days that have no return. Pearson's
r and Spearman's rho are then computed between daily mean sentiment and (a) the
same-day return and (b) the next-day return, each with its p-value and the number
of paired days. Correlation is not computed below three paired days, and is flagged
as indicative only below ten.

**6. Presentation.** A Streamlit interface with two tabs renders summary metrics, a
correlation table with plain-language interpretation, Plotly charts (daily
sentiment timeline, price-versus-sentiment overlay, label distribution) and a
sortable headline table with scores and source links. The FinBERT model is cached
for the session with `st.cache_resource` and all network fetches are cached for one
hour with `st.cache_data`, so the model loads once and repeated interactions do not
re-fetch or re-score.

**7. Text analysis mode.** Pasted text is split into sentences on terminal
punctuation and newlines, fragments shorter than fifteen characters are discarded
as headers or bullets, each sentence is scored with the same model, and the overall
mean score, label breakdown and the five most positive and five most negative
sentences are reported.

**8. Testing.** The aggregation, correlation, deduplication and sentence-splitting
logic is covered by a pytest suite in which the model is replaced by a stub
returning fixed logits, so tests run without downloading weights and verify the
arithmetic rather than the model.

## Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Interface | Streamlit |
| NLP model | FinBERT (`ProsusAI/finbert`) via Hugging Face Transformers |
| Deep learning runtime | PyTorch (CPU) |
| News sources | Google News RSS (feedparser), yfinance news feed |
| Market data | yfinance |
| Analysis | pandas, NumPy, SciPy (`pearsonr`, `spearmanr`) |
| Visualization | Plotly |
| Testing | pytest |

## System architecture

```text
            +------------------------------+
            |   Streamlit UI (app.py)      |
            |   sidebar + two tabs         |
            +--------------+---------------+
                           |
      +--------------------+--------------------+
      |                    |                    |
+-----v------+     +-------v------+     +-------v------+
|  news.py   |     | sentiment.py |     |  prices.py   |
| RSS +      |     | FinBERT      |     | yfinance     |
| yfinance   |     | batched CPU  |     | closes +     |
| dedupe     |     | inference    |     | returns      |
+-----+------+     +-------+------+     +-------+------+
      |                    |                    |
      +--------------------+--------------------+
                           |
                   +-------v--------+      +-----------+
                   |  analysis.py   |----->| charts.py |
                   | daily means,   |      | Plotly    |
                   | join, Pearson, |      +-----------+
                   | Spearman       |
                   +----------------+
```

## Limitations

1. **Short samples.** Free news sources cover only a recent window, so a seven-day
   run typically produces single-digit paired trading days. The application always
   reports `n` and warns below ten paired days; a correlation over so few
   observations is descriptive, not evidence of a predictive relationship.
2. **Headlines rather than article bodies.** Both feeds supply titles only, so the
   classifier never sees the full text of a story.
3. **Day-level alignment.** Sentiment is bucketed by UTC calendar day, so whether
   an article was published before or after the market close is ignored. The effect
   is largest for NSE tickers, because IST leads UTC by five and a half hours.
4. **Weekend and holiday news is excluded** from the correlation, since there is no
   same-day return to pair it against.
5. **Domain mismatch within finance.** FinBERT was fine-tuned on financial phrases
   and analyst language, not news headlines, which are written to attract clicks;
   sarcasm and forward-looking statements can flip a label. A concrete failure
   observed during testing: the sentence "Net liabilities increased modestly as we
   drew on the revolving credit facility" was classified positive with a score of
   +0.905, apparently because of the word "increased". The domain-specific model
   therefore reduces, but does not eliminate, the misreading of financial
   vocabulary that motivated the project.
6. **No causal claim and no backtest.** The application measures association on a
   small sample; it does not test a trading strategy.
7. **Uneven coverage.** Google News indexes US large-caps far more densely than
   NSE mid-caps, so results are weaker for thinly covered Indian stocks.

## Future scope

1. **Full-article text** via an article-extraction library, so the model scores the
   body rather than the headline.
2. **Intraday alignment**, timestamping each article against the relevant
   exchange's session so pre-close and post-close news are separated, with
   market-local time zones instead of UTC bucketing.
3. **Longer history** from an archival news dataset, which would make the
   correlation statistically meaningful instead of indicative, and allow a lagged
   cross-correlation across several days.
4. **Event-study methodology**: measure abnormal returns against a market index
   rather than raw returns, isolating the stock-specific reaction.
5. **Weighting by source credibility and article reach**, instead of treating every
   headline equally in the daily mean.
6. **Comparison against baselines** (VADER, a general-purpose BERT, a bag-of-words
   classifier) on a labelled financial dataset, to quantify how much the
   domain-specific model actually gains.
7. **Fine-tuning** FinBERT on headlines from the Indian market, where vocabulary and
   phrasing differ from the US-centric corpus it was trained on.
8. **Portfolio view**: sentiment across a watchlist, with sector aggregation.

## Conclusion

FinSentiment demonstrates that a domain-specific transformer can be assembled with
free, keyless data sources into a working analytical tool that both measures the
sentiment of financial news and tests it against the market's actual response. The
implementation deliberately reports its own sample size and refuses to overstate a
correlation computed over a handful of trading days, which is itself part of the
lesson: the modelling is the easy half, and the honest evaluation of a small,
noisy, freely-available sample is the harder one.
