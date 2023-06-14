import os.path

from matgl.config import clear_cache, MATGL_CACHE


def test_clear_cache():
    clear_cache(False)
    assert not os.path.exists((MATGL_CACHE))
