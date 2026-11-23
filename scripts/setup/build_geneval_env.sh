#!/bin/bash
# GenEval environment (env_geneval/): the official GenEval detector stack, pinned as in
# external/geneval/environment.yml (mmdetection branch 2.x). Used for the training reward (sprout/geneval_worker.py)
# and for scoring (eval/geneval_score.py). Needs Python 3.10, a CUDA 11.3-compatible driver, and external/ from
# scripts/setup/clone_external.sh (the Mask2Former config is read from external/mmdetection).
# mmdet 3.x must not be used: it renames the Mask2Former head weights, so the v2 checkpoint would load with randomly
# initialized decoder weights and only a warning.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3.10 -m venv env_geneval
source env_geneval/bin/activate
pip install --upgrade "pip<25" "setuptools<81" wheel
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# numpy < 1.24: mmdet 2.x uses the np.float / np.int aliases.
pip install "numpy==1.23.1" "scipy==1.9.1" "pillow==9.4.0" "opencv-python==4.6.0.66" "pycocotools==2.0.6" \
    "transformers==4.36.1" "tokenizers==0.15.0" "open-clip-torch==2.20.0" "clip-benchmark==1.4.0" "openmim==0.3.7" \
    "yapf==0.33.0" "terminaltables==3.1.10" "matplotlib==3.7.1" "tqdm==4.65.0"
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install "mmengine==0.7.3" "mmdet==2.28.2"     # 2.28.2 is the tip of branch 2.x
grep -q "2.28.2" external/mmdetection/mmdet/version.py     # the installed package and the config agree
bash external/geneval/evaluation/download_models.sh external/geneval/weights
python -c "import mmdet; assert mmdet.__version__ == '2.28.2'; from mmdet.apis import init_detector"
