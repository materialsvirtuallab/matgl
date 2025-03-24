#!/bin/bash

uname
python -c 'import dgl; print(f"DGL v{dgl.__version__}")'
python -c 'import torch; print(f"Torch v{torch.__version__}")'
python -c 'import torchdata; print(f"torchdata v{torchdata.__version__}")'

pytest tests
