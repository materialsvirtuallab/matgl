from __future__ import annotations

import os

import requests

from matgl.config import PRETRAINED_MODELS_PATH


class ModelSource:
    """
    Defines a local or remote source for models.
    """

    def __init__(self, uri):
        """

        Args:
            uri:
        """
        self.uri = uri
        self.model_name = uri.split("/")[-1]
        self.local_path = PRETRAINED_MODELS_PATH / self.model_name
        self._download()

    def _download(self):
        r = requests.get(self.uri, allow_redirects=True)

        if not PRETRAINED_MODELS_PATH.exists():
            os.makedirs(PRETRAINED_MODELS_PATH)

        with open(self.local_path, "wb") as f:
            f.write(r.content)
        return self.local_path

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

        Args:
            exc_type:
            exc_val:
            exc_tb:

        Returns:

        """
        self.stream.close()
