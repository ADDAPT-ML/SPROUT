"""GenEval images for a checkpoint, as the official external/geneval/generation/diffusers_generate.py generates
them (553 prompts, 4 images each, 50 steps, guidance 9, seed 42 re-set for every prompt, one image per call,
default safety checker, attention slicing), with the LoRA checkpoint loaded into the UNet.

usage: python -m eval.geneval_generate --lora_dir <run>/lora_iteration_<t> --outdir <dir>   (--lora_dir base: SD1.5)
Then score with eval/geneval_score.py.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from eval.lora import load_lora


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', default='external/geneval/prompts/evaluation_metadata.jsonl')
    p.add_argument('--model', default='runwayml/stable-diffusion-v1-5')
    p.add_argument('--lora_dir', default='base')
    p.add_argument('--outdir', required=True)
    a = p.parse_args()
    n_samples, steps, scale, seed = 4, 50, 9.0, 42
    metadatas = [json.loads(line) for line in open(a.metadata) if line.strip()]
    model = StableDiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.float16).to('cuda')
    model.enable_attention_slicing()
    model.set_progress_bar_config(disable=True)
    if a.lora_dir != 'base':
        model.unet = load_lora(model.unet, a.lora_dir)
    for index, metadata in enumerate(metadatas):
        seed_everything(seed)
        outpath = os.path.join(a.outdir, f'{index:0>5}')
        sample_path = os.path.join(outpath, 'samples')
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, 'metadata.jsonl'), 'w') as fp:
            json.dump(metadata, fp)
        if len(os.listdir(sample_path)) >= n_samples:    # resumable by whole prompts
            continue
        with torch.no_grad():
            for k in range(n_samples):
                image = model(metadata['prompt'], num_inference_steps=steps, guidance_scale=scale,
                              num_images_per_prompt=1).images[0]
                image.save(os.path.join(sample_path, f'{k:05}.png'))


if __name__ == '__main__':
    main()
