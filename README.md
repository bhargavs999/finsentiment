# FinSentiment

Score financial news with **FinBERT** and compare the sentiment against what the
stock price actually did.

Financial news moves prices, but reading it manually is slow, and general-purpose
sentiment models misread finance language: "liability", "write-down" and "beat
estimates" all mean something specific here. FinBERT is trained on financial text,
so it reads those terms the way an analyst would.

## Credits

This project is a fork of
**[eyllasmin/fin-sentiment-tracker](https://github.com/eyllasmin/fin-sentiment-tracker)**
by [eyllasmin](https://github.com/eyllasmin), used under the **MIT License** (the
upstream `LICENSE` file is kept intact).

Kept from upstream: the FinBERT scoring approach, including the compound score
`P(positive) - P(negative)` and the Streamlit + Plotly dashboard shape.

What I added:

- Google News RSS + yfinance news fetching with no API keys (upstream scraped Finviz)
- NSE ticker support (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`) alongside US tickers
- Pearson and Spearman correlation of daily sentiment against same-day and
  next-day returns, with sample size and small-sample caveats
- Batched FinBERT inference with 512-token truncation, cached with `st.cache_resource`
- Daily sentiment aggregation joined onto trading days via yfinance prices
- A second tab that scores pasted text sentence by sentence
- Dedupe across both news sources by normalized title
- A pytest suite (model mocked, no weight downloads) and project docs
- Removed upstream's SQLite persistence and Finviz scraper, which this version
  does not need

## Features

- **Ticker input** for US (`AAPL`, `MSFT`, `NVDA`) and NSE (`RELIANCE.NS`,
  `TCS.NS`, `INFY.NS`) symbols, with presets and a 7 / 14 / 30-day window.
- **News without API keys** from Google News RSS (query built from the ticker plus
  the company name resolved through yfinance) and `yfinance.Ticker.news`, deduped
  by normalized title.
- **FinBERT** (`ProsusAI/finbert`) on CPU, batched, truncated at 512 tokens. Each
  article gets a label, all three class probabilities, and a score of
  `P(positive) - P(negative)`.
- **Correlation** of daily mean sentiment against same-day and next-day returns,
  Pearson and Spearman, always reported with the sample size `n`.
- **Charts** (Plotly): sentiment timeline, price-vs-sentiment overlay, label
  distribution, and a sortable headline table with scores and links.
- **Analyze Text tab**: paste a transcript or press release, get an overall score,
  a label breakdown, and the five most positive and five most negative sentences.

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/bhargavs999/finsentiment.git
cd finsentiment

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. The first run downloads the FinBERT
weights (about 440 MB) and caches them under `~/.cache/huggingface`; later runs
load from that cache.

Run the tests with:

```bash
pytest -q
```

The tests mock the model, so they never download weights.

## Limitations

- **Short sample by construction.** Free news sources only cover a recent window,
  so a 7-day run typically leaves a handful of paired trading days. The app shows
  `n` and warns when the sample is small; a correlation over that few days is not
  evidence that sentiment predicts price.
- **Headlines, not full articles.** Both feeds give titles, so the model scores a
  headline rather than the body of the story.
- **FinBERT is not infallible.** It reduces this class of error rather than
  removing it. In testing it scored *"Net liabilities increased modestly as we drew
  on the revolving credit facility"* as **positive (+0.905)**, apparently keying on
  "increased". The sample text in the Analyze Text tab keeps that sentence so the
  failure is visible rather than hidden.
- **Day-level alignment only.** Sentiment is aggregated per UTC calendar day and
  joined onto trading days, which ignores whether an article landed before or
  after the close. NSE tickers are affected most, since IST sits well ahead of UTC.
- **Correlation is not causation**, and nothing here is backtested.
- Google News coverage of NSE mid-caps is thinner than for US large-caps.
- No database: everything is fetched live and held in Streamlit's cache for an hour.

## Stack

Python 3.11, Streamlit, Transformers, PyTorch (CPU), pandas, NumPy, Plotly,
yfinance, feedparser, SciPy, pytest. Versions are pinned in `requirements.txt`.

## Deploying to Hugging Face Spaces later

Not deployed yet. When you want to:

1. Create a Space at <https://huggingface.co/new-space> and pick the **Streamlit** SDK.
2. Add this repo as a remote and push:
   `git remote add space https://huggingface.co/spaces/<user>/finsentiment && git push space main`
3. Spaces installs `requirements.txt` automatically and runs `app.py`. The free CPU
   tier is enough; the first boot downloads the FinBERT weights, so expect a slow
   cold start.

## Docs

- [`docs/PRD.md`](docs/PRD.md) - problem, requirements, architecture, data sources,
  limitations, decisions log.
- [`docs/PROJECT_REPORT_NOTES.md`](docs/PROJECT_REPORT_NOTES.md) - abstract,
  objectives, methodology and future scope, for the submission write-up.

## License

MIT, inherited from the upstream project. See [`LICENSE`](LICENSE).

---

Educational project. Not financial advice.
