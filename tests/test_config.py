from __future__ import annotations

import os.path

import pytest

import matgl
from matgl.config import MATGL_CACHE, clear_cache


def test_clear_cache():
    clear_cache(False)
    assert not os.path.exists(MATGL_CACHE)


def test_set_backend():
    with pytest.raises(ValueError, match="Invalid backend"):
        matgl.set_backend("nonsense")
