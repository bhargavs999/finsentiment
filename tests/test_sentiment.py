"""Sentence splitting and FinBERT scoring, with the model mocked.

No weights are downloaded here: a fake tokenizer/model pair returns fixed logits so
the scoring arithmetic and batching are tested on their own.
"""

import pytest
import torch

from finsentiment import sentiment


class FakeTokenizer:
    """Records how it was called and returns a dummy batch encoding."""

    def __init__(self):
        self.calls = []

    def __call__(self, batch, **kwargs):
        self.calls.append({"batch": list(batch), "kwargs": kwargs})
        return {"input_ids": torch.zeros(len(batch), 4, dtype=torch.long)}


class FakeModel:
    """Returns one preset logit row per text, in order."""

    def __init__(self, logit_rows, labels=("positive", "negative", "neutral")):
        self.logit_rows = [list(r) for r in logit_rows]
        self.cursor = 0
        self.batch_sizes = []
        self.config = type("Cfg", (), {"id2label": {i: l for i, l in enumerate(labels)}})()

    def __call__(self, **inputs):
        size = inputs["input_ids"].shape[0]
        self.batch_sizes.append(size)
        rows = self.logit_rows[self.cursor : self.cursor + size]
        self.cursor += size
        return type("Out", (), {"logits": torch.tensor(rows, dtype=torch.float)})()


def test_score_texts_computes_label_and_compound_score():
    # Logits strongly favour "positive" on the first row, "negative" on the second.
    model = FakeModel([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    out = sentiment.score_texts(["good", "bad", "meh"], FakeTokenizer(), model)

    assert [r["label"] for r in out] == ["positive", "negative", "neutral"]
    assert out[0]["score"] == pytest.approx(out[0]["p_positive"] - out[0]["p_negative"], abs=1e-4)
    assert out[0]["score"] > 0.9
    assert out[1]["score"] < -0.9
    assert abs(out[2]["score"]) < 0.1


def test_score_texts_probabilities_sum_to_one():
    model = FakeModel([[1.0, 0.5, 0.2]])
    row = sentiment.score_texts(["x"], FakeTokenizer(), model)[0]
    total = row["p_positive"] + row["p_negative"] + row["p_neutral"]

    assert total == pytest.approx(1.0, abs=1e-3)


def test_score_texts_respects_batch_size():
    model = FakeModel([[1.0, 0.0, 0.0]] * 5)
    out = sentiment.score_texts(["a", "b", "c", "d", "e"], FakeTokenizer(), model, batch_size=2)

    assert len(out) == 5
    assert model.batch_sizes == [2, 2, 1]


def test_score_texts_truncates_at_512_tokens():
    tokenizer = FakeTokenizer()
    sentiment.score_texts(["x"], tokenizer, FakeModel([[1.0, 0.0, 0.0]]))
    kwargs = tokenizer.calls[0]["kwargs"]

    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 512


def test_score_texts_on_empty_input():
    assert sentiment.score_texts([], FakeTokenizer(), FakeModel([])) == []


def test_label_order_follows_model_config():
    model = FakeModel([], labels=("negative", "neutral", "positive"))
    assert sentiment.label_order(model) == ["negative", "neutral", "positive"]


def test_label_order_falls_back_when_config_missing():
    assert sentiment.label_order(object()) == sentiment.FALLBACK_LABELS


def test_score_texts_uses_config_label_order():
    # Same logits as the first test, but the model's labels are in a different order.
    model = FakeModel([[5.0, 0.0, 0.0]], labels=("negative", "neutral", "positive"))
    row = sentiment.score_texts(["text"], FakeTokenizer(), model)[0]

    assert row["label"] == "negative"
    assert row["score"] < -0.9


def test_split_sentences_splits_on_terminators_and_newlines():
    text = "Revenue rose 12 percent this quarter. Margins compressed slightly.\nGuidance was raised for the year."
    out = sentiment.split_sentences(text)

    assert len(out) == 3
    assert out[0] == "Revenue rose 12 percent this quarter."
    assert out[2].startswith("Guidance was raised")


def test_split_sentences_drops_short_fragments():
    out = sentiment.split_sentences("Q3. Revenue grew strongly across all segments.")

    assert out == ["Revenue grew strongly across all segments."]


def test_split_sentences_collapses_whitespace():
    out = sentiment.split_sentences("The  company   reported\t a  record  quarter overall.")
    assert out == ["The company reported a record quarter overall."]


def test_split_sentences_on_empty_text():
    assert sentiment.split_sentences("") == []
    assert sentiment.split_sentences(None) == []


def test_split_sentences_caps_count():
    text = " ".join(["This is a long enough sentence to count."] * 50)
    assert len(sentiment.split_sentences(text, max_sentences=10)) == 10
