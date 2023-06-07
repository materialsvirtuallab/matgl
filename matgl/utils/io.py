"""
Provides utilities for managing models and data.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
from pathlib import Path

import requests
import torch

from matgl.config import MATGL_CACHE, PRETRAINED_MODELS_BASE_URL

logger = logging.getLogger(__file__)


class IOMixIn:
    """
    Mixin class for model saving and loading.
    """

    def save_args(self, locals: dict, kwargs: dict) -> None:
        r"""
        Method to save args into a private _init_args variable. This should be called after super in the __init__
        method, e.g., `self.save_args(locals(), kwargs)`.

        Args:
            locals: The result of locals().
            kwargs: kwargs passed to the class.
        """
        args = inspect.getfullargspec(self.__class__.__init__).args
        self._init_args = {k: v for k, v in locals.items() if k in args and k not in ("self", "__class__")}
        self._init_args.update(kwargs)

    def save(self, path: str | Path = ".", metadata: dict | None = None, makedirs: bool = True):
        """
        Save model to a directory. Three files will be saved.
        - path/model.pt, which contains the torch serialized model args.
        - path/state.pt, which contains the saved state_dict from the model.
        - path/model.json, a txt version of model.pt that is purely meant for ease of reference.

        Args:
            path: String or Path object to directory for model saving. Defaults to current working directory (".").
            metadata: Any additional metadata to be saved into the model.json file. For example, a good use would be
                a description of model purpose, the training set used, etc.
            makedirs: Whether to create the directory using os.makedirs(exist_ok=True). Note that if the directory
                already exists, makedirs will not do anything.
        """
        path = Path(path)
        if makedirs:
            os.makedirs(path, exist_ok=True)
        torch.save(self._init_args, path / "model.pt")  # type: ignore
        torch.save(self.state_dict(), path / "state.pt")  # type: ignore

        # This json dump of model args is purely for ease of reference. It is not used to deserialize the model.
        d = {"name": self.__class__.__name__, "metadata": metadata, "kwargs": self._init_args}  # type: ignore
        with open(path / "model.json", "w") as f:
            json.dump(d, f, default=lambda o: str(o), indent=4)

    @classmethod
    def load(cls, path: str | Path, include_json=False, **kwargs):
        """
        Load the model weights from a directory.

        Args:
            path (str|path): Path to saved model or name of pre-trained model. The search order is
                path, followed by model name in PRETRAINED_MODELS_PATH, followed by download from
                PRETRAINED_MODELS_BASE_URL.
            include_json (bool): If True, the "model.json" file is also loaded. This file can contain metadata about
                the model, e.g., if scaling was performed in training the model, this file may contain the mean and
                standard deviation of the models, which needs to be applied to the final predictions.
            kwargs: Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
                want to update the model.

        Returns: model_object if include_json is false. (model_object, dict) if include_json is True.
        """
        path = Path(path)

        fnames = ["model.pt", "state.pt"]
        if include_json:
            fnames.append("model.json")

        if all((path / fn).exists() for fn in fnames):
            fpaths = {fn: path / fn for fn in fnames}
        elif all((MATGL_CACHE / path / fn).exists() for fn in fnames):
            fpaths = {fn: MATGL_CACHE / path / fn for fn in fnames}
        else:
            try:
                fpaths = {
                    fn: RemoteFile(f"{PRETRAINED_MODELS_BASE_URL}{path}/{fn}", **kwargs).local_path for fn in fnames
                }
            except BaseException:
                raise ValueError(
                    f"No valid model found in {path} or among pre-trained_models at "
                    f"{MATGL_CACHE} or {PRETRAINED_MODELS_BASE_URL}."
                ) from None

        if not torch.cuda.is_available():
            state = torch.load(fpaths["state.pt"], map_location=torch.device("cpu"))
        else:
            state = torch.load(fpaths["state.pt"])
        model = cls(**torch.load(fpaths["model.pt"]))
        model.load_state_dict(state)  # type: ignore

        if include_json:
            with open(fpaths["model.json"]) as f:
                model_data = json.load(f)

            return model, model_data

        return model


class RemoteFile:
    """
    Handling of download of remote files to a local cache.
    """

    def __init__(self, uri: str, cache_location: str | Path = MATGL_CACHE, force_download: bool = False):
        """

        Args:
            uri: Uniform resource identifier.
            cache_location: Directory to cache downloaded RemoteFile. By default, downloaded models are saved at
                $HOME/.matgl.
            force_download: To speed up access, a model with the same name in the cache location will be used if
                present. If you want to force a re-download, set this to True.
        """
        self.uri = uri
        toks = uri.split("/")
        self.model_name = toks[-2]
        self.fname = toks[-1]
        cache_location = Path(cache_location)
        os.makedirs(cache_location / self.model_name, exist_ok=True)
        self.local_path = cache_location / self.model_name / self.fname
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
        self.stream = open(self.local_path, "rb")  # noqa: SIM115
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the with context.

        Args:
            exc_type: Usual meaning in __exit__.
            exc_val: Usual meaning in __exit__.
            exc_tb: Usual meaning in __exit__.
        """
        self.stream.close()
