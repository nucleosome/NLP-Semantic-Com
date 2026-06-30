import pytest
from unittest.mock import MagicMock, patch
import torch


def make_mock_inference():
    """Return a DeepSCInference with mocked model and vocab."""
    vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3, "hello": 4, "world": 5}

    with patch("src.inference.torch.load", return_value={}), \
         patch("src.inference.DeepSC") as MockDeepSC, \
         patch("builtins.open", create=True), \
         patch("json.load", return_value=vocab):

        mock_model = MagicMock()
        MockDeepSC.return_value = mock_model

        from src.inference import DeepSCInference
        inference = DeepSCInference.__new__(DeepSCInference)
        inference.vocab = vocab
        inference.vocab_size = len(vocab)
        inference.pad_idx = 0
        inference.start_idx = 1
        inference.end_idx = 2
        inference.channel = "AWGN"
        inference.n_var = 0.1
        inference.model = mock_model

        from utils import SeqtoText
        inference.seq2text = SeqtoText(vocab, 2)

    return inference


def test_tokenize_basic():
    inf = make_mock_inference()
    tensor = inf._tokenize("hello world")
    assert tensor.shape[0] == 1
    assert tensor[0, 0].item() == 1   # <START>
    assert tensor[0, -1].item() == 2  # <END>
    assert tensor[0, 1].item() == 4   # hello
    assert tensor[0, 2].item() == 5   # world


def test_tokenize_unknown_word():
    inf = make_mock_inference()
    tensor = inf._tokenize("unknownword")
    assert tensor[0, 1].item() == 3  # <UNK>


def test_tokenize_empty_raises():
    inf = make_mock_inference()
    # empty string → only start+end tokens, should not crash
    tensor = inf._tokenize("")
    assert tensor.shape[1] == 2  # just <START> <END>


def test_predict_returns_string():
    inf = make_mock_inference()
    output_tensor = torch.tensor([[1, 4, 5, 2]])  # <START> hello world <END>

    with patch("src.inference.greedy_decode", return_value=output_tensor):
        result = inf.predict("hello world")
    assert isinstance(result, str)
    assert len(result) > 0
