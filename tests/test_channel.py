import math
import torch
import pytest
from utils import Channels, SNR_to_noise


@pytest.fixture
def channels():
    return Channels()


def test_snr_to_noise():
    noise = SNR_to_noise(10)
    expected = 1 / math.sqrt(2 * 10 ** (10 / 10))
    assert abs(noise - expected) < 1e-9


def test_awgn_shape(channels):
    sig = torch.zeros(4, 10, 16)
    out = channels.AWGN(sig, n_var=0.1)
    assert out.shape == sig.shape


def test_awgn_adds_noise(channels):
    sig = torch.zeros(100, 10, 16)
    out = channels.AWGN(sig, n_var=1.0)
    assert not torch.allclose(out, sig)


def test_rayleigh_shape(channels):
    sig = torch.zeros(4, 10, 16)
    out = channels.Rayleigh(sig, n_var=0.1)
    assert out.shape == sig.shape


def test_rician_shape(channels):
    sig = torch.zeros(4, 10, 16)
    out = channels.Rician(sig, n_var=0.1, K=1)
    assert out.shape == sig.shape


@pytest.mark.parametrize("snr", [0, 5, 10, 18])
def test_noise_decreases_with_higher_snr(snr):
    n1 = SNR_to_noise(snr)
    n2 = SNR_to_noise(snr + 1)
    assert n2 < n1
