from __future__ import annotations

import os.path

from matgl.config import MATGL_CACHE, clear_cache


def test_clear_cache():
    clear_cache(False)
    assert not os.path.exists(MATGL_CACHE)
