from __future__ import annotations

import os


def test_entrypoint(BaNiO3):
    BaNiO3.to(filename="BaNiO3.cif")
    exit_status = os.system("mgl relax -i BaNiO3.cif -o BaNiO3_relaxed.cif")
    assert exit_status == 0
    assert os.path.exists("BaNiO3_relaxed.cif")
    exit_status = os.system("mgl predict -i BaNiO3.cif -m MEGNet-MP-2018.6.1-Eform")
    assert exit_status == 0
    os.remove("BaNiO3.cif")
    os.remove("BaNiO3_relaxed.cif")
