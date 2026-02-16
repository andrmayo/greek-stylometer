"""Tests for JSONL schema round-trips."""

import json

from greek_stylometer.schemas import (
    ActivationManifestEntry,
    Explanation,
    FeatureWeight,
    Passage,
    Prediction,
)


def test_passage_round_trip():
    p = Passage(
        text="Εἰ μὲν μηδόλως",
        author="Galen",
        author_id="tlg0057",
        work_id="tlg001",
        passage_idx=0,
        source="first1kgreek",
        metadata={"edition": "1st1K-grc1"},
    )
    line = p.to_json()
    p2 = Passage.from_json(line)
    assert p2 == p
    # Verify JSON is valid and has expected keys
    d = json.loads(line)
    assert d["author"] == "Galen"
    assert d["metadata"]["edition"] == "1st1K-grc1"


def test_passage_unicode_preserved():
    p = Passage(
        text="τοῦ κατὰ τὴν ψυχὴν",
        author="Γαληνός",
        author_id="tlg0057",
        work_id="tlg001",
        passage_idx=0,
        source="test",
    )
    line = p.to_json()
    assert "τοῦ" in line  # ensure_ascii=False
    p2 = Passage.from_json(line)
    assert p2.text == "τοῦ κατὰ τὴν ψυχὴν"


def test_prediction_round_trip():
    p = Prediction(
        text="some text",
        author="Galen",
        label=1,
        predicted=1,
        confidence=0.97,
        split="test",
        passage_idx=42,
    )
    assert Prediction.from_json(p.to_json()) == p


def test_explanation_round_trip():
    e = Explanation(
        passage_idx=0,
        text="some text",
        predicted=1,
        confidence=0.95,
        top_features=[
            FeatureWeight(token="τοῦ", weight=0.15),
            FeatureWeight(token="καὶ", weight=-0.08),
        ],
    )
    e2 = Explanation.from_json(e.to_json())
    assert e2.passage_idx == e.passage_idx
    assert len(e2.top_features) == 2
    assert e2.top_features[0].token == "τοῦ"
    assert e2.top_features[1].weight == -0.08


def test_activation_manifest_round_trip():
    a = ActivationManifestEntry(
        passage_idx=0,
        layer=11,
        cls_embedding_file="activations/0_layer11_cls.npy",
        token_embeddings_file="activations/0_layer11_tokens.npy",
    )
    assert ActivationManifestEntry.from_json(a.to_json()) == a


def test_activation_manifest_optional_tokens():
    a = ActivationManifestEntry(
        passage_idx=0,
        layer=11,
        cls_embedding_file="activations/0_layer11_cls.npy",
    )
    a2 = ActivationManifestEntry.from_json(a.to_json())
    assert a2.token_embeddings_file is None
