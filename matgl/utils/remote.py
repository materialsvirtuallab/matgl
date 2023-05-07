"""
Provides utilities for managing models and data.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

from matgl.config import MATGL_CACHE


class RemoteFile:
    """
    Handling of download of remote files to a local cache.
    """

    def __init__(self, uri: str, use_cache: bool = True, force_download: bool = False):
        """

        Args:
            uri: Uniform resource identifier.
            use_cache: By default, downloaded models are saved at $HOME/.matgl. If False, models will be
                downloaded to current working directory.
            force_download: To speed up access, a model with the same name in the cache location will be used if
                present. If you want to force a re-download, set this to True.
        """
        self.uri = uri
        toks = uri.split("/")
        self.model_name = toks[-2]
        self.fname = toks[-1]
        if use_cache:
            os.makedirs(MATGL_CACHE / self.model_name, exist_ok=True)
            self.local_path = MATGL_CACHE / self.model_name / self.fname
        else:
            os.makedirs(self.model_name, exist_ok=True)
            self.local_path = Path(self.model_name) / self.fname
        if (not self.local_path.exists()) or force_download:
            self._download()

    def _download(self):
        r = requests.get(self.uri, allow_redirects=True)
        with open(self.local_path, "wb") as f:
            f.write(r.content)

    def __enter__(self):
        """
        Support with context.

        Returns:
            Stream on local path.
        """
        self.stream = open(self.local_path, "rb")
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the with context.

        Args:
            exc_type:
            exc_val:
            exc_tb:

        Returns:

        """
        self.stream.close()
