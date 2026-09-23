#!/bin/bash
set -e

mamba env create -f environment.yml
mamba run -n skfin_2026 pip install -e .
mamba run -n skfin_2026 python -m ipykernel install --user --name skfin_2026 --display-name "Python (skfin)"
