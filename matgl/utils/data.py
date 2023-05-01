from __future__ import annotations

import requests


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
        self.local_path = self._download()

    def _download(self):
        name = self.uri.split("/")[-1]
        r = requests.get(self.uri, allow_redirects=True)
        with open(name, "wb") as f:
            f.write(r.content)
        return name

    def __enter__(self):
        """
        Support with context.

        Returns:

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
