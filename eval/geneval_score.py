"""Score GenEval images with the official evaluate_images.py, after checking that every key of the detector
checkpoint matches the configured model (a mismatched checkpoint otherwise loads with only a warning).
Run in the GenEval environment:

  env_geneval/bin/python eval/geneval_score.py <image dir> --outfile <results.jsonl> \
      --model-path external/geneval/weights \
      --model-config external/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py
  env_geneval/bin/python external/geneval/evaluation/summary_scores.py <results.jsonl>
"""
import runpy
import sys
from pathlib import Path

import mmdet.apis
import torch

_init_detector = mmdet.apis.init_detector


def checked_init_detector(config, checkpoint, *args, **kwargs):
    detector = _init_detector(config, checkpoint, *args, **kwargs)
    state = torch.load(checkpoint, map_location='cpu')
    if set(detector.state_dict()) != set(state.get('state_dict', state)):
        raise RuntimeError('detector checkpoint keys do not match the configured model')
    return detector


mmdet.apis.init_detector = checked_init_detector
target = Path(__file__).resolve().parents[1] / 'external' / 'geneval' / 'evaluation' / 'evaluate_images.py'
sys.argv[0] = str(target)
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name='__main__')
