from __future__ import annotations

import os
import unittest

import torch

from matgl.utils.data import ModelSource


class ModelSourceTestCase(unittest.TestCase):
    def test_remote(self):
        with ModelSource(
            "https://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/MEGNet-MP-2018.6.1-Eform.pt",
            use_cache=False,
        ) as s:
            model = torch.load(s, map_location=torch.device("cpu"))
            self.assertIn("state_dict", model["model"])
        try:  # cleanup
            os.remove("MEGNet-MP-2018.6.1-Eform.pt")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
