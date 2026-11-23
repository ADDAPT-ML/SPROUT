"""DPG-Bench images for a checkpoint, in the layout of the official scorer (external/ELLA/dpg_bench): 4 images
per prompt of dpg_bench/prompts/, tiled 2x2 under the prompt's file name. The release fixes no sampler, so all
checkpoints use: 512 x 512, the pipeline's default scheduler, 50 steps, guidance 7.5, image k seeded 42 + k,
no safety checker.

usage: python -m eval.dpg_generate --lora_dir <run>/lora_iteration_<t> --outdir <dir>
       bash external/ELLA/dpg_bench/dist_eval.sh <dir> 512
"""
import argparse
import os

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from eval.lora import load_lora


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompts_dir', default='external/ELLA/dpg_bench/prompts')
    p.add_argument('--lora_dir', default='base')
    p.add_argument('--outdir', required=True)
    a = p.parse_args()
    size, k_images, steps, cfg, seed = 512, 4, 50, 7.5, 42
    files = sorted(f for f in os.listdir(a.prompts_dir) if f.endswith('.txt'))
    os.makedirs(a.outdir, exist_ok=True)
    pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16,
                                                   safety_checker=None, feature_extractor=None,
                                                   requires_safety_checker=False).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    if a.lora_dir != 'base':
        pipe.unet = load_lora(pipe.unet, a.lora_dir)
    for f in files:
        out = os.path.join(a.outdir, f.replace('.txt', '.png'))
        if os.path.exists(out):
            continue
        prompt = open(os.path.join(a.prompts_dir, f)).read().strip()
        with torch.no_grad():
            images = pipe([prompt] * k_images, height=size, width=size, num_inference_steps=steps, guidance_scale=cfg,
                          generator=[torch.Generator('cuda').manual_seed(seed + k) for k in range(k_images)]).images
        grid = Image.new('RGB', (2 * size, 2 * size))
        for k, image in enumerate(images):
            grid.paste(image, ((k % 2) * size, (k // 2) * size))
        grid.save(out + '.tmp.png')
        os.replace(out + '.tmp.png', out)


if __name__ == '__main__':
    main()
