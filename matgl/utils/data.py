from __future__ import annotations

import requests


class ModelSource:
    def __init__(self, uri):
        """

        Args:
            uri:
        """
        self.uri = uri

    def _download(self):
        r = requests.get(self.uri, allow_redirects=True)
        with open("facebook.ico", "wb") as f:
            f.write(r.content)

    def __enter__(self):
        """

        Returns:

        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """

        Args:
            exc_type:
            exc_val:
            exc_tb:

        Returns:

        """
        pass
