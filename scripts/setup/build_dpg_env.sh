#!/bin/bash
# DPG-Bench scorer environment (env_dpg/): mPLUG-large VQA through modelscope, from external/ELLA's requirements.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3.10 -m venv env_dpg
V=env_dpg/bin
$V/pip install "pip==23.3.2" "setuptools<70" wheel cython numpy==1.26.4
$V/pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
$V/pip install -r external/ELLA/requirements-for-dpg_bench.txt
# modelscope's framework dependencies, which ELLA's requirements do not pull
$V/pip install "numpy==1.26.4" "transformers==4.40.2" "huggingface_hub==0.25.2" "accelerate==0.30.1" "diffusers==0.27.2" \
    addict attrs einops python-dateutil "PyYAML>=5.4" scipy "simplejson>=3.3.0" "sortedcontainers>=1.5.9" ftfy \
    "datasets>=4.0.0,<=4.8.4"
