"""Tests for the preprocessing pipeline."""

import pytest
from unittest.mock import patch, MagicMock


def test_load_config(tmp_path):
    from src.preprocess import load_config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("project:\n  name: test\n")
    config = load_config(str(config_file))
    assert config["project"]["name"] == "test"


def test_build_label_map():
    from src.preprocess import build_label_map

    mock_features = MagicMock()
    mock_features.__getitem__.return_value = MagicMock(names=["booking", "weather", "cancel"])

    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = MagicMock(features=mock_features)

    label2id, id2label = build_label_map(mock_dataset)

    assert label2id["booking"] == 0
    assert label2id["weather"] == 1
    assert id2label[0] == "booking"
    assert len(label2id) == 3


def test_label_map_roundtrip():
    """Ensure label2id and id2label are consistent inverses."""
    from src.preprocess import build_label_map

    labels = ["intent_a", "intent_b", "intent_c"]
    mock_features = MagicMock()
    mock_features.__getitem__.return_value = MagicMock(names=labels)

    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = MagicMock(features=mock_features)

    label2id, id2label = build_label_map(mock_dataset)

    for label, idx in label2id.items():
        assert id2label[idx] == label
