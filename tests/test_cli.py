from __future__ import annotations

import os
from pathlib import Path


def test_entrypoint(Mo):
    Mo.to(filename="Mo.cif")
    exit_status = os.system("mgl relax -i Mo.cif -o Mo_relaxed.cif")
    assert exit_status == 0
    exit_status = os.system("mgl relax -i Mo.cif -s _hello")
    assert exit_status == 0
    assert os.path.exists("Mo_hello.cif")
    exit_status = os.system("mgl relax -i Mo.cif")
    assert exit_status == 0
    exit_status = os.system("mgl predict -i Mo.cif -s 1 -m MEGNet-MP-2019.4.1-BandGap-mfi")
    assert exit_status == 0
    exit_status = os.system("mgl predict -i Mo.cif -m MEGNet-MP-2018.6.1-Eform")
    assert exit_status == 0

    # if "PMG_MAPI_KEY" in SETTINGS:
    #     exit_status = os.system("mgl predict -p mp-19017 -m MEGNet-MP-2018.6.1-Eform")
    #     assert exit_status == 0
    exit_status = os.system("mgl clear --yes")
    assert exit_status == 0
    assert not (Path(os.path.expanduser("~")) / ".cache/matgl").exists()
    os.remove("Mo.cif")
    os.remove("Mo_relaxed.cif")
    os.remove("Mo_hello.cif")
