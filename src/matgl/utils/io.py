"""Provides utilities for managing models and data."""

from __future__ import annotations

import inspect
import json
import logging
import os
import warnings
from pathlib import Path
from urllib.parse import urlparse

import fsspec
import requests
import torch

from matgl.config import MATGL_CACHE, PRETRAINED_MODELS_BASE_URL

logger = logging.getLogger(__file__)


def ropen(uri: str | Path, mode="rb", *, cache_location=MATGL_CACHE, **kwargs):
    """
    Open a file with fsspec, using a cache for remote URLs, and local file access otherwise.

    Args:
        uri (str): The URI or local file path.
        mode (str): File mode, e.g., "rb", "r".
        cache_location (str): Local directory for caching remote files.
        **kwargs: Extra arguments passed to fsspec.

    Returns:
        A file-like object.
    """
    parsed = urlparse(str(uri))
    is_remote = parsed.scheme in ("http", "https", "s3", "ftp")

    if is_remote:
        # We implement a specialized cache naming system to make it easier for people to locate the model weights.
        toks = str(uri).split("/")
        model_name = toks[-2]
        cache_location = Path(cache_location)

        fs = fsspec.filesystem(
            "filecache",
            target_protocol=parsed.scheme,
            cache_storage=str(cache_location / model_name),
            same_names=True,
        )
        return fs.open(uri, mode, **kwargs)
    return open(Path(uri), mode, **kwargs)


class IOMixIn:
    """Mixin class for model saving and loading.

    For proper usage, models should subclass nn.Module and IOMix and the `save_args` method should be called
    immediately after the `super().__init__()` call::

        super().__init__()
        self.save_args(locals(), kwargs)

    """

    def save_args(self, locals: dict, kwargs: dict | None = None) -> None:
        r"""Method to save args into a private _init_args variable.

        This should be called after super in the __init__ method, e.g., `self.save_args(locals(), kwargs)`.

        Args:
            locals: The result of locals().
            kwargs: kwargs passed to the class.
        """
        args = inspect.getfullargspec(self.__class__.__init__).args
        d = {k: v for k, v in locals.items() if k in args and k not in ("self", "__class__")}
        if kwargs is not None:
            d.update(kwargs)

        # If one of the args is a subclass of IOMixIn, we will serialize that class.
        for k, v in d.items():
            if issubclass(v.__class__, IOMixIn):
                d[k] = {
                    "@class": v.__class__.__name__,
                    "@module": v.__class__.__module__,
                    "@model_version": getattr(v, "__version__", 0),
                    "init_args": v._init_args,
                }
        self._init_args = d

    def save(self, path: str | Path = ".", metadata: dict | None = None, makedirs: bool = True):
        """Save model to a directory.

        Three files will be saved.
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
        d = {
            "@class": self.__class__.__name__,
            "@module": self.__class__.__module__,
            "@model_version": getattr(self, "__version__", 0),
            "metadata": metadata,
            "kwargs": self._init_args,
        }  # type: ignore
        with open(path / "model.json", "w") as f:
            json.dump(d, f, default=lambda o: str(o), indent=4)

    @classmethod
    def load(cls, path: str | Path | dict, **kwargs):
        """Load the model weights from a directory.

        Args:
            path (str|path|dict): Path to saved model or name of pre-trained model. If it is a dict, it is assumed to
                be of the form::

                    {
                        "model.pt": path to model.pt file,
                        "state.pt": path to state file,
                        "model.json": path to model.json file
                    }

                Otherwise, the search order is path, followed by download from PRETRAINED_MODELS_BASE_URL
                (with caching).
            **kwargs: Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
                want to update the model.

        Returns: model_object.
        """
        fpaths = path if isinstance(path, dict) else _get_file_paths(Path(path), **kwargs)

        with ropen(fpaths["model.json"], "rt") as f:
            model_data = json.load(f)

        _check_ver(cls, model_data)

        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        with ropen(fpaths["state.pt"], "rb") as f:
            state = torch.load(f, map_location=map_location)
        with ropen(fpaths["model.pt"], "rb") as f:
            d = torch.load(f, map_location=map_location)

        # Deserialize any args that are IOMixIn subclasses.
        for k, v in d.items():
            if isinstance(v, dict) and "@class" in v and "@module" in v:
                modname = v["@module"]
                classname = v["@class"]
                mod = __import__(modname, globals(), locals(), [classname], 0)
                cls_ = getattr(mod, classname)
                _check_ver(cls_, v)  # Check version of any subclasses too.
                d[k] = cls_(**v["init_args"])
        d = {k: v for k, v in d.items() if not k.startswith("@")}
        model = cls(**d)
        model.load_state_dict(state, strict=False)  # type: ignore

        return model


def load_model(path: str | Path, **kwargs):
    r"""Convenience method to load a model from a directory or name.

    Args:
        path (str|path): Path to saved model or name of pre-trained model. The search order is path, followed by
            download from PRETRAINED_MODELS_BASE_URL (with caching).
        **kwargs: Additional kwargs passed to RemoteFile class. E.g., a useful one might be force_download if you
            want to update the model.

    Returns:
        Returns: model_object if include_json is false. (model_object, dict) if include_json is True.
    """
    path = Path(path)

    fpaths = _get_file_paths(path, **kwargs)

    try:
        with ropen(fpaths["model.json"], "rt") as f:
            d = json.load(f)
            modname = d["@module"]
            classname = d["@class"]

            mod = __import__(modname, globals(), locals(), [classname], 0)
            cls_ = getattr(mod, classname)
            return cls_.load(fpaths, **kwargs)
    except BaseException as err:
        raise ValueError(
            "Bad serialized model or bad model name. It is possible that you have an older model cached. Please "
            'clear your cache by running `python -c "import matgl; matgl.clear_cache()"`'
        ) from err


def _get_file_paths(path: Path):
    """Search path for files.

    Args:
        path (Path): Path to saved model or name of pre-trained model. The search order is path, followed by
            download from PRETRAINED_MODELS_BASE_URL (with caching).

    Returns:
        {
            "model.pt": path to model.pt file,
            "state.pt": path to state file,
            "model.json": path to model.json file
        }
    """
    fnames = ("model.pt", "state.pt", "model.json")

    if all((path / fn).exists() for fn in fnames):
        return {fn: path / fn for fn in fnames}

    try:
        return {fn: f"{PRETRAINED_MODELS_BASE_URL}{path}/{fn}" for fn in fnames}
    except requests.RequestException:
        raise ValueError(f"No valid model found in pre-trained_models at {PRETRAINED_MODELS_BASE_URL}.") from None


def _check_ver(cls_, d: dict):
    """Check version of cls_ in current matgl against those noted in a model.json dict.

    Args:
        cls_: Class object.
        d: Dict from serialized json.

    Raises:
        Deprecation warning if the code is
    """
    if getattr(cls_, "__version__", 0) > d.get("@model_version", 0):
        warnings.warn(
            "Incompatible model version detected! The code will continue to load the model but it is "
            "recommended that you provide a path to an updated model, increment your @model_version in model.json "
            "if you are confident that the changes are not problematic, or clear your ~/.matgl cache using "
            '`python -c "import matgl; matgl.clear_cache()"`',
            UserWarning,
            stacklevel=2,
        )


def get_available_pretrained_models() -> list[str]:
    """Checks Github for available pretrained_models for download. These can be used with load_model.

    Returns:
        List of available models.
    """
    try:
        r = requests.get("https://api.github.com/repos/materialsvirtuallab/matgl/contents/pretrained_models")
        return [d["name"] for d in json.loads(r.content.decode("utf-8")) if d["type"] == "dir"]
    except Exception:
        print("Unable to access GitHub to check for pre-trained models.")
        return []
