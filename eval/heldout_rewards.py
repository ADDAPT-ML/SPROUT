"""Score GenEval images with four reward models that never enter training, each through its authors' code:

  pickscore    external/PickScore README, "Inference with PickScore": the README's code is executed as written,
               except that calc_probs also returns the scores (logit_scale.exp() * cosine) next to the probabilities.
  hps          HPS v2.1 through the official hpsv2 package (hpsv2.img_score.score with hps_version='v2.1').
  aesthetic    LAION improved-aesthetic-predictor, external/improved-aesthetic-predictor/simple_inference.py: its
               MLP, normalization, model loading and per-image statements are executed from the file unchanged
               (pytorch_lightning.LightningModule is bound to torch.nn.Module, which leaves the model unchanged).
  imagereward  the official image-reward package: RM.load and model.score.

usage: python -m eval.heldout_rewards <GenEval image dir> --out <scores.jsonl>
Writes one line per prompt with the per-image scores, and <scores>.summary.json with each model's mean.
"""
import argparse
import ast
import io
import json
import math
import os
import shutil
import types
from pathlib import Path

EXTERNAL = Path(__file__).resolve().parents[1] / 'external'
MODELS = ('pickscore', 'hps', 'aesthetic', 'imagereward')


class PickScore:
    def __init__(self):
        text = (EXTERNAL / 'PickScore' / 'README.md').read_text()
        start = text.index('```python', text.index('## Inference with PickScore')) + len('```python')
        src = text[start:text.index('```', start)]
        old = 'return probs.cpu().tolist()'
        assert src.count(old) == 1, 'unexpected PickScore README'
        tree = ast.parse(src.replace(old, 'return scores.cpu().tolist(), probs.cpu().tolist()'))
        # keep the imports, the model loading and calc_probs; drop the README's example call
        end = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef) and n.name == 'calc_probs')
        tree.body = tree.body[:end + 1]
        self.ns = {}
        exec(compile(tree, 'PickScore/README.md', 'exec'), self.ns)

    def __call__(self, prompt, images):
        scores, probs = self.ns['calc_probs'](prompt, images)
        z = [math.exp(s - max(scores)) for s in scores]
        assert all(abs(p - e / sum(z)) < 1e-4 for p, e in zip(probs, z))
        return scores


class Aesthetic:
    def __init__(self):
        import clip
        import numpy as np
        import torch
        import torch.nn as nn
        from PIL import Image
        root = EXTERNAL / 'improved-aesthetic-predictor'
        body = [n for n in ast.parse((root / 'simple_inference.py').read_text()).body
                if not isinstance(n, (ast.Import, ast.ImportFrom))]
        src = [ast.unparse(n) for n in body]
        setup = next(i for i, s in enumerate(src) if s.startswith('model = MLP(768)'))
        image = next(i for i, s in enumerate(src) if s.startswith('pil_image = Image.open(img_path)'))
        end = next(i for i, s in enumerate(src) if s.startswith('prediction = model('))
        defs = [n for n in body[:setup] if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        assert {n.name for n in defs} == {'MLP', 'normalized'}

        def module(nodes):
            return compile(ast.Module(body=nodes, type_ignores=[]), 'simple_inference.py', 'exec')
        self.ns = dict(np=np, torch=torch, clip=clip, Image=Image, nn=nn,
                       pl=types.SimpleNamespace(LightningModule=nn.Module))
        exec(module(defs), self.ns)
        cwd = os.getcwd()
        os.chdir(root)      # the script loads its weights by a relative path
        try:
            exec(module(body[setup:image]), self.ns)
        finally:
            os.chdir(cwd)
        self.per_image = module(body[image:end + 1])

    def __call__(self, pngs):
        out = []
        for data in pngs:
            self.ns['img_path'] = io.BytesIO(data)
            exec(self.per_image, self.ns)
            out.append(float(self.ns['prediction'].item()))
        return out


class HeldOutRewards:
    def __init__(self):
        import huggingface_hub
        import hpsv2.img_score
        import ImageReward as RM
        import torch
        self.torch, self.hps_score = torch, hpsv2.img_score.score
        self.pickscore, self.aesthetic = PickScore(), Aesthetic()
        # hpsv2 loads its checkpoint on every call; read it from a local copy.
        self.hps_checkpoint = os.path.join(os.environ.get('TMPDIR', '/tmp'), f'hps_{os.getpid()}.pt')
        shutil.copy(huggingface_hub.hf_hub_download('xswu/HPSv2', 'HPS_v2.1_compressed.pt'), self.hps_checkpoint)
        self.imagereward = RM.load('ImageReward-v1.0', device='cuda')

    def __call__(self, prompt, pngs):
        from PIL import Image
        images = [Image.open(io.BytesIO(b)).convert('RGB') for b in pngs]
        hps = [float(x) for x in self.hps_score(images, prompt, cp=self.hps_checkpoint, hps_version='v2.1')]
        with self.torch.no_grad():
            ir = self.imagereward.score(prompt, images)
        ir = ir if isinstance(ir, list) else [ir]     # a one-image list is returned as a scalar
        return dict(pickscore=self.pickscore(prompt, images), hps=hps, aesthetic=self.aesthetic(pngs),
                    imagereward=[float(x) for x in ir])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image_dir', help='GenEval layout: <index>/metadata.jsonl and <index>/samples/*.png')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rewards = HeldOutRewards()
    rows = []
    for folder in sorted(os.listdir(a.image_dir)):
        d = Path(a.image_dir) / folder
        if not (d / 'metadata.jsonl').is_file():
            continue
        meta = json.loads((d / 'metadata.jsonl').read_text().splitlines()[0])
        pngs = [(d / 'samples' / f).read_bytes() for f in sorted(os.listdir(d / 'samples')) if f.endswith('.png')]
        rows.append(dict(folder=folder, prompt=meta['prompt'], tag=meta.get('tag', ''), **rewards(meta['prompt'], pngs)))
    with open(a.out, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    mean = {m: sum(sum(r[m]) for r in rows) / sum(len(r[m]) for r in rows) for m in MODELS}
    Path(a.out + '.summary.json').write_text(json.dumps(dict(n_prompts=len(rows), mean=mean), indent=1))
    print(mean)
    os.remove(rewards.hps_checkpoint)


if __name__ == '__main__':
    main()
