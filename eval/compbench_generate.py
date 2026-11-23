"""T2I-CompBench++ images for a checkpoint and a category, in the layout of the official scorers. Follows
external/T2I-CompBench/GORS_finetune/inference_eval.py (prompts: first field of examples/dataset/<category>_val.txt;
10 images per prompt; DDIM from the model's scheduler config, 30 steps, guidance 7.5, 512 px, fp16, default safety
checker; image n of a prompt seeded 42 + n; files <prompt>_<count:06>.png), with SD1.5 and our LoRA loading.

usage: python -m eval.compbench_generate --category <category> --lora_dir <run>/lora_iteration_<t> --outdir <dir>/<category>
Then run the category's official scorer in external/T2I-CompBench on <dir>/<category>.
"""
import argparse
import os

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from eval.lora import load_lora


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--category', required=True)
    p.add_argument('--model_id', default='runwayml/stable-diffusion-v1-5')
    p.add_argument('--lora_dir', default='base')
    p.add_argument('--outdir', required=True, help='images go to <outdir>/samples')
    p.add_argument('--dataset_dir', default='external/T2I-CompBench/examples/dataset')
    a = p.parse_args()
    n_images, steps, seed = 10, 30, 42
    with open(os.path.join(a.dataset_dir, f'{a.category}_val.txt')) as f:
        prompts = [line.split('\t')[0] for line in f.read().splitlines()]
    samples = os.path.join(a.outdir, 'samples')
    os.makedirs(samples, exist_ok=True)
    scheduler = DDIMScheduler.from_pretrained(a.model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(a.model_id, scheduler=scheduler, torch_dtype=torch.float16).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    if a.lora_dir != 'base':
        pipe.unet = load_lora(pipe.unet, a.lora_dir)
    for i, prompt in enumerate(prompts):
        paths = [os.path.join(samples, f'{prompt}_{i * n_images + n:06}.png') for n in range(n_images)]
        if all(os.path.exists(x) for x in paths):
            continue
        with torch.no_grad():
            out = pipe([prompt] * n_images, num_inference_steps=steps,
                       generator=[torch.Generator('cuda').manual_seed(seed + n) for n in range(n_images)])
        for x, image in zip(paths, out.images):
            image.save(x + '.tmp.png')
            os.replace(x + '.tmp.png', x)


if __name__ == '__main__':
    main()
