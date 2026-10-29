"""FinBERT scoring on CPU.

Scoring follows the upstream project: a compound score of P(positive) - P(negative),
so one number spans -1 (clearly bad news) to +1 (clearly good news).
"""

from __future__ import annotations

import re

MODEL_NAME = "ProsusAI/finbert"
MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 16

# ProsusAI/finbert's own label order; used only if the model config lacks id2label.
FALLBACK_LABELS = ["positive", "negative", "neutral"]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS_RE = re.compile(r"\s+")


def load_model(model_name: str = MODEL_NAME):
    """Load tokenizer and model on CPU in eval mode. Downloads weights on first call."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def label_order(model) -> list[str]:
    """Label names in logit order, read from the model config when available."""
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if isinstance(id2label, dict) and id2label:
        try:
            return [str(id2label[i]).lower() for i in sorted(id2label, key=int)]
        except (KeyError, ValueError, TypeError):
            pass
    return list(FALLBACK_LABELS)


def score_texts(texts, tokenizer, model, batch_size: int = DEFAULT_BATCH_SIZE):
    """Score texts with FinBERT.

    Returns one dict per text: label, p_positive, p_negative, p_neutral and
    score = P(positive) - P(negative).
    """
    import torch

    texts = [str(t) for t in texts]
    if not texts:
        return []

    labels = label_order(model)
    results = []

    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start : start + max(1, batch_size)]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        for row in probs:
            values = [float(v) for v in row]
            by_label = dict(zip(labels, values))
            p_pos = by_label.get("positive", 0.0)
            p_neg = by_label.get("negative", 0.0)
            p_neu = by_label.get("neutral", 0.0)
            best = labels[max(range(len(values)), key=lambda i: values[i])]
            results.append(
                {
                    "label": best,
                    "p_positive": round(p_pos, 4),
                    "p_negative": round(p_neg, 4),
                    "p_neutral": round(p_neu, 4),
                    "score": round(p_pos - p_neg, 4),
                }
            )
    return results


def split_sentences(text: str, min_chars: int = 15, max_sentences: int = 300):
    """Split a transcript or press release into sentences worth scoring.

    Fragments shorter than min_chars are dropped: headers and list bullets add noise
    without carrying sentiment.
    """
    if not text:
        return []
    chunks = _SENTENCE_SPLIT_RE.split(str(text))
    out = []
    for chunk in chunks:
        cleaned = _WS_RE.sub(" ", chunk).strip()
        if len(cleaned) >= min_chars:
            out.append(cleaned)
        if len(out) >= max_sentences:
            break
    return out
