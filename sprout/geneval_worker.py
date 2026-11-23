#!/usr/bin/env python3
"""GenEval reward server, run by the trainer as a subprocess in the GenEval environment (env_geneval).

The official GenEval detector needs python 3.10 / torch 1.12.1 / mmcv-full 1.7.1 / mmdet 2.28.2, which cannot
share a process with the training environment. The worker speaks JSON lines:
  in : {"image": "<png path>", "meta": <GenEval metadata record>}
  out: {"reward": <partial credit in [0, 1]>, "correct": 0|1}
  in : {"cmd": "quit"}

`correct` is the official all-requirements verdict of evaluate_image. `reward` is the fraction of the prompt's
requirements that the official evaluate_image accepts when called with the metadata reduced to that single
requirement, on the same detections. A position requirement keeps the object it refers to, so that the
relation is checked against it as in the full evaluation.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# Run as a script, this file's directory heads sys.path; drop it so that the package's module names (data,
# reward, ...) cannot shadow third-party ones.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != HERE]
sys.path.insert(0, os.path.join(REPO, 'external', 'geneval', 'evaluation'))


def build(model_config, model_path):
    """Import the official evaluate_images module and set the globals its __main__ block sets."""
    import evaluate_images as EI

    class Args:
        pass
    args = Args()
    args.model_config, args.model_path, args.options = model_config, model_path, {}
    EI.args = args
    detector, (clip_model, transform, tokenizer), classnames = EI.load_models(args)
    EI.object_detector = detector
    EI.clip_model, EI.transform, EI.tokenizer = clip_model, transform, tokenizer
    EI.classnames = classnames
    EI.THRESHOLD = 0.3
    EI.COUNTING_THRESHOLD = 0.9
    EI.MAX_OBJECTS = 16
    EI.NMS_THRESHOLD = 1.0
    EI.POSITION_THRESHOLD = 0.1
    # Detections depend only on the image, so the per-requirement calls reuse the first detection.
    raw, cache = EI.inference_detector, {'key': None, 'value': None}

    def cached(model, filepath, *a, **kw):
        if cache['key'] != filepath:
            cache['key'], cache['value'] = filepath, raw(model, filepath, *a, **kw)
        return cache['value']
    EI.inference_detector = cached
    return EI


def score(EI, image_path, meta):
    """Return (partial-credit reward, official verdict)."""
    correct = 1 if EI.evaluate_image(image_path, meta)['correct'] else 0
    include, exclude = list(meta.get('include', [])), list(meta.get('exclude', []))
    requirements = [('include', i) for i in range(len(include))] + [('exclude', i) for i in range(len(exclude))]
    if not requirements:
        return float(correct), correct
    satisfied = 0
    for kind, index in requirements:
        sub = dict(meta)
        if kind == 'include':
            req = include[index]
            position = req.get('position')
            if position is not None:
                anchor = include[position[1]]
                sub['include'] = [anchor, dict(req, position=[position[0], 0])]
            else:
                sub['include'] = [req]
            sub['exclude'] = []
        else:
            sub['include'], sub['exclude'] = [], [exclude[index]]
        if EI.evaluate_image(image_path, sub)['correct']:
            satisfied += 1
    return satisfied / len(requirements), correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    a = parser.parse_args()
    if a.deterministic:
        sys.path.insert(0, REPO)
        from sprout.determinism import configure
        configure(a.seed, mask2former=True)
    # mmdet logs to stdout; keep stdout for the protocol only.
    protocol, sys.stdout = sys.stdout, sys.stderr

    def emit(obj):
        protocol.write(json.dumps(obj) + '\n')
        protocol.flush()

    EI = build(a.model_config, a.model_path)
    emit({'ready': True})
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get('cmd') == 'quit':
            break
        try:
            reward, correct = score(EI, request['image'], request['meta'])
            emit({'reward': reward, 'correct': correct})
        except Exception as e:  # noqa: BLE001 -- reported to the trainer, which fails
            emit({'error': f'{type(e).__name__}: {e}'})


if __name__ == '__main__':
    main()
