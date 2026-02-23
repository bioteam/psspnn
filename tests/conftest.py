"""Shared pytest fixtures for psspnn tests."""

import numpy as np
import pytest

from psspnn.model.network import HolleyKarplusNet


@pytest.fixture
def tiny_sequence() -> str:
    """A short amino acid sequence for unit tests."""
    return "ACDEFGHIKL"


@pytest.fixture
def tiny_ss() -> list[str]:
    return ["H", "H", "H", "H", "E", "E", "C", "C", "C", "C"]


@pytest.fixture
def net_2h() -> HolleyKarplusNet:
    """Default 2-hidden-unit network with fixed seed."""
    return HolleyKarplusNet(window_size=5, hidden_units=2, seed=42)


@pytest.fixture
def net_0h() -> HolleyKarplusNet:
    """Zero-hidden-unit network with fixed seed."""
    return HolleyKarplusNet(window_size=5, hidden_units=0, seed=42)
