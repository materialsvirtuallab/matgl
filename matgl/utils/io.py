"""
Provides utilities for managing models and data.
"""
from __future__ import annotations

import json
import logging
import os
from inspect import getfullargspec
from pathlib import Path

import requests
import torch

from matgl.config import MATGL_CACHE, PRETRAINED_MODELS_BASE_URL

logger = logging.getLogger(__file__)


def _get_class_args(obj):
    d = {}
    spec = getfullargspec(obj.__class__.__init__)
    args = spec.args

    for c in args:
        if c != "self":
            try:
                a = getattr(obj, c)
            except AttributeError:
                raise NotImplementedError(
                    "All args to be present as either self.argname and kwargs to be present under "
                    "a self.kwargs variable."
                )
            d[c] = a
    return d


class IOMixIn:
    """
    Mixin class for model saving and loading.
    """

    def save(self, path: str | Path, metadata: dict | None = None):
        """
        Save model to a directory. Three files will be saved.
        - path/model.pt, which contains the torch serialzied model args.
        - path/state.pt, which contains the saved state_dict from the model.
        - path/model.txt, a txt version of model.pt that is purely meant for ease of reference.

        Args:
            path: String or Path object to directory for model saving.
            metadata: Any additional metadata to be saved into the model.txt file. For example, a good use would be
                a description of model purpose, the training set used, etc.
        """
        path = Path(path)
        model_args = _get_class_args(self)
        torch.save(model_args, path / "model.pt")  # type: ignore
        torch.save(self.state_dict(), path / "state.pt")  # type: ignore

        # This txt dump of model args is purely for ease of reference. It is not used to deserialize the model.
        d = {"name": self.__class__.__name__, "metadata": metadata, "kwargs": model_args}  # type: ignore
        with open(path / "model.txt", "w") as f:
            json.dump(d, f, default=lambda o: str(o), indent=4)

    @classmethod
    def load(cls, path: str | Path):
        """
        Load the model weights from a directory.

        Args:
            path (str|path): Path to saved model or name of pre-trained model. The search order is
                path, followed by model name in PRETRAINED_MODELS_PATH, followed by download from
                PRETRAINED_MODELS_BASE_URL.

        Returns: MEGNet object.
        """
        path = Path(path)
        if (path / "model.pt").exists() and (path / "state.pt").exists():
            model_path = path / "model.pt"
            state_path = path / "state.pt"
        elif (MATGL_CACHE / path / "model.pt").exists() and (MATGL_CACHE / path / "state.pt").exists():
            model_path = MATGL_CACHE / path / "model.pt"
            state_path = MATGL_CACHE / path / "state.pt"
        else:
            try:
                model_file = RemoteFile(f"{PRETRAINED_MODELS_BASE_URL}{path}/model.pt")
                state_file = RemoteFile(f"{PRETRAINED_MODELS_BASE_URL}{path}/state.pt")
                model_path = model_file.local_path
                state_path = state_file.local_path
            except BaseException:
                raise ValueError(
                    f"No valid model found in {model_path} or among pre-trained_models at "
                    f"{MATGL_CACHE} or {PRETRAINED_MODELS_BASE_URL}."
                )

        if not torch.cuda.is_available():
            state = torch.load(state_path, map_location=torch.device("cpu"))
        else:
            state = torch.load(state_path)
        model_args = torch.load(model_path)
        model = cls(**model_args)
        model.load_state_dict(state)  # type: ignore
        return model


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
            logger.info("Downloading from remote location...")
            self._download()
        else:
            logger.info(f"Using cached local file at {self.local_path}...")

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
