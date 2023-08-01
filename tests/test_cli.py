from __future__ import annotations

import os


def test_entrypoint(Mo):
    Mo.to(filename="Mo.cif")
    exit_status = os.system("mgl relax -i Mo.cif -o Mo_relaxed.cif")
    assert exit_status == 0
    exit_status = os.system("mgl relax -i Mo.cif -s _hello")
    assert exit_status == 0
    assert os.path.exists("Mo_hello.cif")
    exit_status = os.system("mgl relax -i Mo.cif")
    assert exit_status == 0
    exit_status = os.system("mgl predict -i Mo.cif -m MEGNet-MP-2018.6.1-Eform")
    assert exit_status == 0
    os.remove("Mo.cif")
    os.remove("Mo_relaxed.cif")
    os.remove("Mo_hello.cif")
