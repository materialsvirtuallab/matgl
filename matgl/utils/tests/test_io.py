from __future__ import annotations

import shutil
import unittest

import torch

from matgl.utils.io import RemoteFile


class ModelSourceTestCase(unittest.TestCase):
    def test_remote(self):
        with RemoteFile(
            "https://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/MEGNet-MP-2018.6.1-Eform/model.pt",
            cache_location=".",
        ) as s:
            d = torch.load(s, map_location=torch.device("cpu"))
            assert "nblocks" in d["model"]["init_args"]
        try:  # cleanup
            shutil.rmtree("MEGNet-MP-2018.6.1-Eform")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
