#!/bin/bash
# T2I-CompBench++ scorer environment (env_compbench/): BLIP-VQA, UniDet (detectron2 @ 5aeb252), CLIPScore and 3-in-1,
# pinned by external/T2I-CompBench/requirements.txt (torch 2.0.1, CUDA 11.7), with the diffusers bundled in the repo.
# Needs Python 3.10, CUDA 11.7 (nvcc) and gcc 9 to build detectron2.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=external/T2I-CompBench
python3.10 -m venv env_compbench
V=$PWD/env_compbench/bin
$V/pip install "pip==23.3.2" "setuptools<70" wheel
$V/pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
grep -v -e detectron2 -e '^nvidia-' -e '^torch' -e '^triton' -e '^lit==' $C/requirements.txt > /tmp/compbench_requirements.txt
$V/pip install -r /tmp/compbench_requirements.txt
tmp=$(mktemp -d) && (cd $tmp && unzip -q "$OLDPWD/$C/diffusers.zip" && $V/pip install ./diffusers)
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")") FORCE_CUDA=1
$V/pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13"
# weights the T2I-CompBench README asks to download: a UniDet expert (Google Drive) and BLIP-VQA (into the timm cache)
$V/pip install gdown
(cd $C/UniDet_eval/experts/expert_weights && $V/gdown "https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq")
(cd $C/BLIPvqa_eval && $V/python -c "from timm.models.hub import download_cached_file; download_cached_file(
  'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth', check_hash=False)")
