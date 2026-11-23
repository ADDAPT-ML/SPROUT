# SPROUT: Predicting Objective-Grounded Prompt Utility for Efficient Text-to-Image Preference Tuning

SPROUT selects the training prompts of each iteration of text-to-image preference tuning.
After every iteration, each trained prompt is labeled with its update norm: the norm of the gradient that the
preference objective (DGPO) produces from the prompt's group of images. A text-only utility model (ridge regression
on frozen CLIP ViT-L/14 text embeddings) is refit on all labels so far and predicts the utility of every prompt in
the pool, and rejection sampling turns the predictions into the next batch. No images are generated beyond those
used for training.

## Layout

| Path | Contents |
| --- | --- |
| `sprout/train.py` | Training: iterative DGPO on Stable Diffusion 1.5 with LoRA, with uniform sampling, CGPO or SPROUT selecting the prompts. |
| `sprout/dgpo_loss.py` | DGPO objective, translated from the authors' code. |
| `sprout/selection.py` | Uniform sampling, SPROUT's rejection sampling and top-B, and CGPO (ported from its released code). |
| `sprout/utility.py` | Utility labels (update norm, reward variance), label histories, and the utility models (ridge on CLIP text embeddings; fine-tuned CLIP text tower). |
| `sprout/reward.py`, `sprout/geneval_worker.py` | Training reward: partial credit from the official GenEval detector, which runs in its own environment. |
| `sprout/data.py` | Training prompt pool. |
| `sprout/determinism.py` | Optional bitwise-reproducible execution. |
| `eval/` | Image generation for GenEval, DPG-Bench and T2I-CompBench++, their official scorers, and the held-out reward models. |
| `scripts/setup/` | Environments. |
| `scripts/verify_cgpo_port.py` | Checks the CGPO port against CGPO's released code, executed verbatim. |
| `tests/` | Unit tests. |
| `external/` | Code of the methods and benchmarks used (see below). |

## Setup

Four Python environments are used, because the official scorers pin incompatible versions:

| Environment | Used for | Built with |
| --- | --- | --- |
| `env/` | training, image generation, held-out reward models | `pip install -r scripts/setup/requirements.txt` (Python 3.9) |
| `env_geneval/` | GenEval detector: training reward and GenEval scoring | `scripts/setup/build_geneval_env.sh` |
| `env_dpg/` | DPG-Bench scorer | `scripts/setup/build_dpg_env.sh` |
| `env_compbench/` | T2I-CompBench++ scorers | `scripts/setup/build_compbench_env.sh` |

Run `scripts/setup/clone_external.sh` first. It clones the code of the following, from their public repositories at
the commits used, into `external/`:

- DGPO, whose `dataset/geneval/train_metadata.jsonl` is the training set;
- CGPO;
- GenEval and mmdetection (branch 2.x);
- DPG-Bench (`ELLA`) and T2I-CompBench;
- PickScore, HPS v2, ImageReward and the LAION aesthetic predictor.

Commands are run from the repository root with `PYTHONPATH=.`.

## Training

All methods share the configuration in Table 1 of the supplementary material, which the defaults of
`sprout/train.py` reproduce:

- Stable Diffusion 1.5, LoRA rank 64;
- B = 8 prompts with G = 24 images each per iteration, for 100 iterations;
- constant learning rate 1.65e-4, KL coefficient 0.02, β = 100;
- the 50,000-row GenEval training set released with DGPO.

The methods differ only in how each iteration's prompts are chosen:

```
python -m sprout.train --output_dir <run> --seed <seed> --selection sprout                       # SPROUT
python -m sprout.train --output_dir <run> --seed <seed> --selection sprout --utility variance    # SPROUT w/ variance
python -m sprout.train --output_dir <run> --seed <seed> --selection uniform                      # uniform sampling
python -m sprout.train --output_dir <run> --seed <seed> --selection cgpo                         # CGPO
```

The paper uses seeds 197, 211 and 223 (seeds A, B and C). The ablations and the larger budgets change these options:

| Setting | Options |
| --- | --- |
| Label history | `--label_history latest` or `previous` (default `all`) |
| Batch selection | `--batch_selection top_b` (default `rejection`) |
| B = 16 | `--num_prompts 16` |
| Predictors at B = 16 | `--ridge_alpha 1000` (with `--utility variance`), `--predictor clip_finetune` |
| 200 iterations | rerun a finished 100-iteration run with `--resume --iterations 200 --lr_schedule cosine --lr_decay_start 100 --lr_decay_iters 20 --lr_min_ratio 0.3 --lr_tail_ratio 0.0` |

Each iteration takes about 10 minutes on one V100 (B = 8). A run writes the following to its output directory:

- the selected prompts, rewards and training statistics of every iteration (`iteration_<t>.json`, `rollouts_<t>.jsonl`);
- the evaluated (EMA) LoRA weights `lora_iteration_<t>/`;
- the utility models in `models/`;
- a checkpoint from which `--resume` continues exactly.

## Evaluation

Each checkpoint `<run>/lora_iteration_<t>` (t = 20, 40, ..., 100) is evaluated as follows.

**GenEval** (553 prompts, 4 images each; the official generation settings):

```
python -m eval.geneval_generate --lora_dir <run>/lora_iteration_<t> --outdir <images>
env_geneval/bin/python eval/geneval_score.py <images> --outfile <results.jsonl> --model-path external/geneval/weights \
    --model-config external/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py
env_geneval/bin/python external/geneval/evaluation/summary_scores.py <results.jsonl>
```

**DPG-Bench:**

```
python -m eval.dpg_generate --lora_dir <run>/lora_iteration_<t> --outdir <images>
bash eval/dpg_score.sh <images> <results.txt>
```

**T2I-CompBench++,** for each of the eight categories (`color`, `shape`, `texture`, `spatial`, `non_spatial`,
`complex`, `numeracy`, `3d_spatial`):

```
python -m eval.compbench_generate --category <category> --lora_dir <run>/lora_iteration_<t> --outdir <dir>/<category>
bash eval/compbench_score.sh <category> <dir>/<category>
```

**Held-out reward models.** The GenEval images are re-scored with PickScore, HPS v2.1, the LAION aesthetic
predictor and ImageReward, each through its authors' code:

```
python -m eval.heldout_rewards <images> --out <scores.jsonl>
```

`--lora_dir base` evaluates Stable Diffusion 1.5 itself (iteration 0).

## Tests

```
python -m pytest tests
python scripts/verify_cgpo_port.py <CGPO run>
```

`scripts/verify_cgpo_port.py` executes the probability update and the sampler of CGPO's released training script
verbatim. It checks that `sprout/selection.py` produces identical probabilities after every iteration of a CGPO run,
and identical batches.
