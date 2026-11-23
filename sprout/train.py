"""Iterative DGPO preference tuning of Stable Diffusion 1.5 with prompt selection.

Each iteration selects B prompts, generates G images per prompt with the current model, scores them with the
partial-credit GenEval reward, and takes one DGPO update. `--selection` chooses how the prompts are selected:

  uniform  uniform sampling of training-set rows (the official DGPO recipe)
  cgpo     CGPO's variance-based rejection sampler (Li et al., CVPR 2026), ported from its released code
  sprout   SPROUT: a text-only utility model, fit on the utility labels of earlier iterations, predicts the
           utility of every prompt in the pool; the batch is drawn by rejection sampling (or top-B)

The DGPO update follows the authors' GenEval recipe (Luo-Yihong/DGPO, commit 6a45077), adapted to SD1.5: the
epsilon-prediction MSE replaces the flow-matching loss, with a 10-step DDIM sampler; four of its first five
timesteps are trained on; the UNet runs in fp16 with dynamic loss scaling. Defaults are the paper's settings.

usage: python -m sprout.train --output_dir <dir> --seed <seed> --selection {uniform,cgpo,sprout} [options]
"""
import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--selection', choices=['uniform', 'cgpo', 'sprout'], required=True)
    p.add_argument('--resume', action='store_true', help='continue from the latest committed iteration')
    p.add_argument('--deterministic', action='store_true', help='bitwise-reproducible execution')
    # SPROUT
    p.add_argument('--utility', choices=['update_norm', 'variance'], default='update_norm',
                   help='utility label: the update norm (SPROUT) or the reward variance (SPROUT w/ variance)')
    p.add_argument('--label_history', choices=['all', 'latest', 'previous'], default='all',
                   help='labels the utility model is fit on: all, the latest per prompt, or the previous iteration')
    p.add_argument('--batch_selection', choices=['rejection', 'top_b'], default='rejection')
    p.add_argument('--predictor', choices=['ridge', 'clip_finetune'], default='ridge')
    p.add_argument('--ridge_alpha', type=float, default=10.0)
    p.add_argument('--clip_finetune_backbone_lr', type=float, default=1e-5)
    p.add_argument('--clip_finetune_head_lr', type=float, default=3e-4)
    p.add_argument('--acceptance_clip', type=float, default=0.6, help='rejection sampling, as in CGPO')
    p.add_argument('--acceptance_floor', type=float, default=0.1, help='rejection sampling, as in CGPO')
    # CGPO
    p.add_argument('--cgpo_category_calibration', action=argparse.BooleanOptionalAction, default=True,
                   help="CGPO's per-task acceptance weights (in its released code)")
    # Data and reward
    p.add_argument('--geneval_metadata', default='external/dgpo/dataset/geneval/train_metadata.jsonl',
                   help='training prompts and their GenEval requirements (released with DGPO)')
    # Training
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--num_prompts', type=int, default=8, help='B, prompts per iteration')
    p.add_argument('--group_size', type=int, default=24, help='G, images per prompt')
    p.add_argument('--model_id', default='runwayml/stable-diffusion-v1-5')
    p.add_argument('--lora_rank', type=int, default=64)
    p.add_argument('--lora_alpha', type=int, default=128)
    p.add_argument('--lr', type=float, default=1.65e-4)
    p.add_argument('--lr_schedule', choices=['constant', 'cosine'], default='constant')
    p.add_argument('--lr_decay_start', type=int, default=0, help='cosine: constant up to this iteration')
    p.add_argument('--lr_decay_iters', type=int, default=0,
                   help='cosine: reach the floor this many iterations after --lr_decay_start (0: at the last iteration)')
    p.add_argument('--lr_min_ratio', type=float, default=0.1, help='cosine: floor as a fraction of --lr')
    p.add_argument('--lr_tail_ratio', type=float, default=None,
                   help='cosine with --lr_decay_iters: after the floor, decay by a second cosine to this fraction '
                        'of --lr at the last iteration')
    p.add_argument('--adam_beta2', type=float, default=0.999)
    p.add_argument('--dgpo_beta', type=float, default=100.0)
    p.add_argument('--kl_beta', type=float, default=0.02)
    p.add_argument('--clip_range', type=float, default=0.01)
    p.add_argument('--num_inference_steps', type=int, default=10)
    p.add_argument('--guidance_scale', type=float, default=4.5)
    p.add_argument('--train_timesteps', type=int, default=4)
    p.add_argument('--trunc_index', type=int, default=4)
    p.add_argument('--ema_decay', type=float, default=0.9)
    p.add_argument('--ema_interval', type=int, default=8)
    p.add_argument('--ema_calls_per_update', type=int, default=None,
                   help="evaluation-EMA steps per update; DGPO steps it once per sampling minibatch of 4 GPUs x 16 "
                        "images, B * G / 64 (default)")
    return p


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.predictor == 'clip_finetune' and args.label_history != 'all':
        parser.error('the fine-tuned CLIP predictor is defined on all labels (--label_history all)')
    if args.ema_calls_per_update is None:
        args.ema_calls_per_update = int(args.num_prompts / (4 * 16 / args.group_size))
    return args


if __name__ == '__main__':
    _early = argparse.ArgumentParser(add_help=False)
    _early.add_argument('--deterministic', action='store_true')
    _early.add_argument('--seed', type=int, default=0)
    _flags, _ = _early.parse_known_args()
    if _flags.deterministic:     # must precede the torch import
        from sprout.determinism import configure
        configure(_flags.seed, argv=['-m', 'sprout.train', *sys.argv[1:]])

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import DDIMScheduler, StableDiffusionPipeline  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

from sprout import selection, utility  # noqa: E402
from sprout.data import distinct, load_pool  # noqa: E402
from sprout.dgpo_loss import UPSTREAM_COMMIT, advantages, group_weight, sample_surrogate  # noqa: E402
from sprout.reward import GenEvalReward, load_metadata  # noqa: E402

# Arguments that may change when a run is resumed (the schedule is checked separately).
RESUMABLE = {'output_dir', 'resume', 'iterations', 'lr_schedule', 'lr_decay_start', 'lr_decay_iters',
             'lr_min_ratio', 'lr_tail_ratio'}


def lr_at(a, iteration):
    """Learning rate of `iteration` (1-based) for the arguments in dict `a`."""
    lr = a['lr']
    if a['lr_schedule'] != 'cosine':
        return lr
    start = int(a['lr_decay_start'])
    if iteration <= start:
        return lr
    first = start + 1 if start else 1
    span = int(a['lr_decay_iters'])
    total = max(1, (start + span if span else int(a['iterations'])) - first)
    frac = min(1.0, max(0.0, (iteration - first) / total))
    lo = lr * float(a['lr_min_ratio'])
    tail = a['lr_tail_ratio']
    if tail is not None and span and iteration > start + span:
        f2 = min(1.0, (iteration - (start + span)) / max(1, int(a['iterations']) - (start + span)))
        end = lr * float(tail)
        return end + 0.5 * (lo - end) * (1 + math.cos(math.pi * f2))
    return lo + 0.5 * (lr - lo) * (1 + math.cos(math.pi * frac))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.dir = Path(args.output_dir)
        (self.dir / 'models').mkdir(parents=True, exist_ok=True)
        set_seed(args.seed)
        self.rows = load_pool(args.geneval_metadata)
        self.prompts = distinct(self.rows)
        self.task_of = None
        if args.selection == 'cgpo' and args.cgpo_category_calibration:
            self.task_of = {p: r['tag'] for p, r in load_metadata(args.geneval_metadata).items()}
        self.iteration = 0
        self.records = []           # utility labels: {prompt, label, iteration}
        self.cgpo_carried = []      # CGPO rows accepted beyond B, used first at the next iteration
        self.pipe = None
        self.reward = None
        self.clip_ft = None
        self.resume_state = None
        if args.resume:
            self._load_checkpoint()

    # -- model -------------------------------------------------------------------------------------------------

    def _ensure_pipeline(self):
        if self.pipe is not None:
            return self.pipe
        a = self.args
        pipe = StableDiffusionPipeline.from_pretrained(a.model_id, torch_dtype=torch.float16,
                                                       safety_checker=None).to('cuda')
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if self.resume_state is None:
            pipe.unet = get_peft_model(pipe.unet, LoraConfig(
                r=a.lora_rank, lora_alpha=a.lora_alpha, target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'],
                lora_dropout=0.0, init_lora_weights='gaussian', bias='none'))
        else:   # live (not EMA) weights of the last committed iteration
            raw = self.dir / f"lora_raw_iteration_{self.resume_state['iteration']}"
            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(raw), is_trainable=True)
        pipe.to('cuda')
        pipe.unet.eval()
        self.pipe = pipe
        self.opt = torch.optim.AdamW(pipe.unet.parameters(), lr=a.lr, betas=(0.9, a.adam_beta2), eps=1e-8,
                                     weight_decay=1e-4)
        self.trainable = {n: p for n, p in pipe.unet.named_parameters() if p.requires_grad}
        self.global_step = 0
        self.ema_old = {n: p.detach().clone() for n, p in self.trainable.items()}    # rollout/clipping reference
        self.ema_eval = {n: p.detach().clone() for n, p in self.trainable.items()}   # evaluated weights
        if self.resume_state is not None:
            s = self.resume_state
            if [n for n, _ in pipe.unet.named_parameters()] != s['optimizer_parameter_names']:
                raise RuntimeError('optimizer parameters changed')
            self.opt.load_state_dict(s['optimizer'])
            self.ema_old = {n: v.to(self.trainable[n]) for n, v in s['ema_old'].items()}
            self.ema_eval = {n: v.to(self.trainable[n]) for n, v in s['ema_eval'].items()}
            self.global_step = s['global_step']
            # Load the auxiliary models before restoring the random state, so that loading them does not
            # advance the restored streams.
            self._reward()
            if a.selection == 'sprout' and a.predictor == 'ridge':
                utility.clip_text_model()
            random.setstate(s['python_random_state'])
            np.random.set_state(s['numpy_random_state'])
            torch.set_rng_state(s['torch_rng_state'])
            torch.cuda.set_rng_state_all(s['cuda_rng_state_all'])
        return pipe

    def _reward(self):
        if self.reward is None:
            self.reward = GenEvalReward(self.args.geneval_metadata, str(self.dir / '.reward_tmp'),
                                        self.args.seed if self.args.deterministic else None)
        return self.reward

    @contextlib.contextmanager
    def _swap_weights(self, weights):
        current = {n: p.detach().clone() for n, p in self.trainable.items()}
        try:
            with torch.no_grad():
                for n, p in self.trainable.items():
                    p.copy_(weights[n])
            yield
        finally:
            with torch.no_grad():
                for n, p in self.trainable.items():
                    p.copy_(current[n])

    @torch.no_grad()
    def _ema_eval_step(self, step):
        """DGPO's EMAModuleWrapper.step: decay min((1+s)/(10+s), 0.9), applied when (s+1) % interval == 0."""
        if (step + 1) % self.args.ema_interval != 0:
            return
        decay = min((1 + step) / (10 + step), self.args.ema_decay)
        for n, p in self.trainable.items():
            self.ema_eval[n].add_((1 - decay) * (p.detach() - self.ema_eval[n]))

    # -- selection ---------------------------------------------------------------------------------------------

    def select(self):
        a, t = self.args, self.iteration
        info = {}
        if a.selection == 'uniform':
            selected = selection.uniform(self.rows, a.num_prompts)
        elif a.selection == 'cgpo':
            selected, self.cgpo_carried = selection.cgpo(
                self.rows, a.num_prompts, selection.load_rounds(self.dir, t), self.cgpo_carried, a.seed, t,
                self.task_of)
        elif t == 1:                # no labels yet: uniform over distinct prompts
            selected = [self.prompts[i] for i in random.sample(range(len(self.prompts)), a.num_prompts)]
        else:
            model = utility.load_utility(self.dir / 'models' / f'utility_iter_{t - 1}.json')
            predictions = np.asarray([model.predict(p) for p in self.prompts], dtype=float)
            if not np.isfinite(predictions).all():
                raise FloatingPointError('non-finite predicted utility')
            if a.batch_selection == 'top_b':
                selected = selection.top_b(self.prompts, predictions, a.num_prompts)
            else:
                selected, probs = selection.rejection_sampling(self.prompts, predictions, a.num_prompts, a.seed, t,
                                                               a.acceptance_clip, a.acceptance_floor)
                info['acceptance_probability'] = {p: float(probs[self.prompts.index(p)]) for p in selected}
            info['predicted_utility'] = {p: float(predictions[self.prompts.index(p)]) for p in selected}
        return selected, info

    # -- rollouts ----------------------------------------------------------------------------------------------

    def rollouts(self, selected):
        """G images per selected prompt from the current policy, with their rewards. A prompt drawn twice gets
        G more images in the same group, as DGPO's per-prompt grouping does."""
        a, pipe, reward = self.args, self.pipe, self._reward()
        torch.cuda.empty_cache()    # the reward worker shares the GPU
        groups = {}
        pipe.unet.eval()
        for prompt in selected:
            g = groups.setdefault(prompt, dict(rewards=[], correct=[], latents=[]))
            for _ in range(a.group_size):
                with torch.no_grad():
                    latent = pipe(prompt, num_inference_steps=a.num_inference_steps, guidance_scale=a.guidance_scale,
                                  eta=0.0, output_type='latent').images
                    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
                    image = pipe.image_processor.postprocess(decoded, output_type='pil')[0]
                r, correct = reward(image, prompt)
                g['rewards'].append(r)
                g['correct'].append(correct)
                g['latents'].append(latent.detach().cpu())
        with open(self.dir / f'rollouts_{self.iteration}.jsonl', 'w') as f:
            for prompt, g in groups.items():
                for i, (r, c) in enumerate(zip(g['rewards'], g['correct'])):
                    f.write(json.dumps(dict(prompt=prompt, image=i, reward=r, correct=c)) + '\n')
        return groups

    # -- DGPO update -------------------------------------------------------------------------------------------

    def update(self, groups):
        """One DGPO update on all groups. Returns the summary and, per prompt, the norm of its contribution to
        the accumulated gradient before clipping (the update norm)."""
        a, pipe, opt = self.args, self.pipe, self.opt
        unet = pipe.unet
        unet.train()
        pipe.scheduler.set_timesteps(a.num_inference_steps, device='cuda')
        indices = random.Random(a.seed * 10007 + self.iteration).sample(range(a.trunc_index + 1), a.train_timesteps)
        timesteps = pipe.scheduler.timesteps[indices]
        normalizer = sum(len(g['rewards']) for g in groups.values()) * len(timesteps)
        opt.zero_grad(set_to_none=True)
        acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.trainable.items()}
        loss_scale, overflows = float(2 ** 16), 0
        track_norm = a.selection == 'sprout' and a.utility == 'update_norm'
        update_norm = {}
        clipping, kls, losses = [], [], []
        active = 0
        for gi, (prompt, g) in enumerate(groups.items()):
            adv = advantages(torch.tensor(g['rewards'], device='cuda', dtype=torch.float32))
            own = {n: torch.zeros_like(v) for n, v in acc.items()} if track_norm else None
            active += int(bool((adv > 0).any() and (adv < 0).any()))
            tokens = pipe.tokenizer(prompt, padding='max_length', truncation=True,
                                    max_length=pipe.tokenizer.model_max_length, return_tensors='pt')
            with torch.no_grad():
                embeds = pipe.text_encoder(tokens.input_ids.to('cuda'))[0]
            latents = g['latents']
            for ti, t in enumerate(timesteps):
                # One noise draw per group and timestep, shared by the group's samples.
                generator = torch.Generator(device='cuda').manual_seed(
                    (a.seed * 1000003 + self.iteration * 10007 + gi * 101 + ti) % (2 ** 31 - 1))
                noise = torch.randn(latents[0].shape, device='cuda', dtype=latents[0].dtype, generator=generator)
                cached, dsms, ref_dsms = [], [], []
                for latent in latents:
                    noisy = pipe.scheduler.add_noise(latent.to('cuda'), noise, t.reshape(1))
                    with torch.no_grad():
                        with unet.disable_adapter():
                            ref = unet(noisy, t, encoder_hidden_states=embeds).sample.float()
                        with self._swap_weights(self.ema_old):
                            old = unet(noisy, t, encoder_hidden_states=embeds).sample.float()
                        pred = unet(noisy, t, encoder_hidden_states=embeds).sample.float()
                        dsms.append((noise.float() - pred).square().mean())
                        ref_dsms.append((noise.float() - ref).square().mean())
                        old_dsm = (noise.float() - old).square().mean()
                    cached.append((noisy, ref, old_dsm))
                weight = group_weight(torch.stack(dsms), torch.stack(ref_dsms), adv, a.dgpo_beta, a.group_size)
                if not torch.isfinite(weight):
                    raise FloatingPointError('non-finite DGPO group weight')
                for adv_i, (noisy, ref, old_dsm) in zip(adv, cached):
                    # fp16 UNet: backpropagate each sample at loss scale S into an fp32 accumulator; on overflow,
                    # halve S and redo the sample. Unscaled, the LoRA gradients underflow to zero.
                    while True:
                        pred = unet(noisy, t, encoder_hidden_states=embeds).sample
                        loss, clipped, kl = sample_surrogate(pred, noise, ref, old_dsm, adv_i, weight,
                                                             a.clip_range, a.kl_beta)
                        (loss * loss_scale).backward()
                        if all(p.grad is None or torch.isfinite(p.grad).all() for p in self.trainable.values()):
                            break
                        opt.zero_grad(set_to_none=True)
                        loss_scale /= 2
                        overflows += 1
                        if loss_scale < 1:
                            raise FloatingPointError('fp16 loss scale fell below 1')
                    with torch.no_grad():
                        for n, p in self.trainable.items():
                            if p.grad is not None:
                                acc[n].add_(p.grad, alpha=1.0 / (loss_scale * normalizer))
                                if own is not None:
                                    own[n].add_(p.grad, alpha=1.0 / (loss_scale * normalizer))
                    opt.zero_grad(set_to_none=True)
                    losses.append(float(loss.detach()))
                    clipping.append(float(clipped))
                    kls.append(float(kl.detach()))
            if own is not None:
                update_norm[prompt] = float(torch.sqrt(sum(v.square().sum() for v in own.values())))
                del own
        for n, p in self.trainable.items():
            p.grad = acc[n]
        if not any(bool((v != 0).any()) for v in acc.values()):
            raise FloatingPointError('all DGPO gradients are zero')
        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.trainable.values()), 1.0, error_if_nonfinite=True)
        # DGPO steps the evaluation EMA once per sampling minibatch: all but the last before the optimizer step.
        for _ in range(a.ema_calls_per_update - 1):
            self._ema_eval_step(self.global_step)
        lr = lr_at(vars(a), self.iteration)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.step()
        decay = min(0.3, 0.001 * self.global_step)
        with torch.no_grad():
            for n, p in self.trainable.items():
                self.ema_old[n].lerp_(p, 1 - decay)
        self.global_step += 1
        self._ema_eval_step(self.global_step)
        unet.eval()
        unet.save_pretrained(self.dir / f'lora_raw_iteration_{self.iteration}')
        with self._swap_weights(self.ema_eval):
            unet.save_pretrained(self.dir / f'lora_iteration_{self.iteration}')     # the evaluated checkpoint
        torch.cuda.empty_cache()
        summary = dict(lr=lr, training_timesteps=[int(t) for t in timesteps], mean_loss=float(np.mean(losses)),
                       mean_kl=float(np.mean(kls)), clip_fraction=float(np.mean(clipping)),
                       grad_norm_before_clip=float(grad_norm), loss_scale=loss_scale, loss_scale_overflows=overflows,
                       active_groups=active, global_step=self.global_step)
        return summary, update_norm

    # -- utility model -----------------------------------------------------------------------------------------

    def fit_utility(self, labels):
        """Add this iteration's labels and refit the utility model that selects the next batch."""
        a, t = self.args, self.iteration
        self.records += [dict(prompt=p, label=y, iteration=t) for p, y in labels]
        rows = utility.label_history(self.records, t, a.label_history)
        prompts, ys = [r['prompt'] for r in rows], [r['label'] for r in rows]
        path = self.dir / 'models' / f'utility_iter_{t}.json'
        if a.predictor == 'ridge':
            utility.RidgeUtility.fit(prompts, ys, a.ridge_alpha).save(path)
            return
        if self.clip_ft is None:
            self.clip_ft = utility.FinetunedCLIPUtility(a.seed, a.clip_finetune_backbone_lr, a.clip_finetune_head_lr)
            if t > 1:   # resumed: continue from the last committed state
                self.clip_ft.load(self.dir / 'models', t - 1)
        self.clip_ft.update(prompts, ys, t)
        weights, optim = self.clip_ft.save(self.dir / 'models', t)
        write_json(path, dict(type='clip_finetune', weights_file=weights, optim_file=optim))

    # -- iteration and checkpoints -----------------------------------------------------------------------------

    def run_iteration(self):
        a = self.args
        self._ensure_pipeline()
        self._reward()
        selected, info = self.select()
        groups = self.rollouts(selected)
        sprout = a.selection == 'sprout'
        if sprout and a.utility == 'variance':
            self.fit_utility([(p, utility.variance_label(groups[p]['rewards'])) for p in selected])
        summary, update_norm = self.update(groups)
        if sprout and a.utility == 'update_norm':
            labels = utility.update_norm_labels(update_norm)
            self.fit_utility([(p, labels[p]) for p in selected])
        return dict(iteration=self.iteration, selected=selected, selection=info,
                    mean_reward={p: float(np.mean(g['rewards'])) for p, g in groups.items()},
                    update_norm=update_norm, dgpo=summary, dgpo_upstream_commit=UPSTREAM_COMMIT)

    def _commit_files(self, t):
        names = [f'iteration_{t}.json', f'state_iteration_{t}.pt']
        for d in (f'lora_iteration_{t}', f'lora_raw_iteration_{t}'):
            names += [f'{d}/adapter_model.safetensors', f'{d}/adapter_config.json']
        model = self.dir / 'models' / f'utility_iter_{t}.json'
        if model.exists():
            names.append(f'models/{model.name}')
            spec = json.loads(model.read_text())
            names += [f"models/{spec[k]}" for k in ('weights_file', 'optim_file') if k in spec]
        return names

    def save_checkpoint(self, result):
        t = self.iteration
        write_json(self.dir / f'iteration_{t}.json', result)
        state = dict(iteration=t, args=vars(self.args), optimizer=self.opt.state_dict(),
                     optimizer_parameter_names=[n for n, _ in self.pipe.unet.named_parameters()],
                     ema_old={n: p.cpu() for n, p in self.ema_old.items()},
                     ema_eval={n: p.cpu() for n, p in self.ema_eval.items()}, global_step=self.global_step,
                     records=self.records, cgpo_carried=self.cgpo_carried,
                     python_random_state=random.getstate(), numpy_random_state=np.random.get_state(),
                     torch_rng_state=torch.get_rng_state(), cuda_rng_state_all=torch.cuda.get_rng_state_all())
        path = self.dir / f'state_iteration_{t}.pt'
        torch.save(state, f'{path}.tmp.{os.getpid()}')
        os.replace(f'{path}.tmp.{os.getpid()}', path)
        # The commit file, written last, marks the iteration complete and records the hash of every file.
        write_json(self.dir / f'commit_{t}.json', {n: sha256(self.dir / n) for n in self._commit_files(t)})

    def _load_checkpoint(self):
        done = [t for t in range(self.args.iterations, 0, -1) if (self.dir / f'commit_{t}.json').is_file()]
        if not done:
            raise RuntimeError(f'--resume: no committed iteration in {self.dir}')
        t = done[0]
        for name, digest in json.loads((self.dir / f'commit_{t}.json').read_text()).items():
            if sha256(self.dir / name) != digest:
                raise RuntimeError(f'{name} changed after iteration {t} was committed')
        state = torch.load(self.dir / f'state_iteration_{t}.pt', map_location='cpu', weights_only=False)
        saved, new = state['args'], vars(self.args)
        changed = [k for k in new if k not in RESUMABLE and saved.get(k) != new[k]]
        if changed:
            raise RuntimeError(f'--resume with changed arguments: {changed}')
        if any(not math.isclose(lr_at(saved, i), lr_at(new, i), rel_tol=1e-12) for i in range(1, t + 1)):
            raise RuntimeError('--resume: the learning rate of completed iterations would change')
        self.iteration, self.resume_state = t, state
        self.records, self.cgpo_carried = state['records'], state['cgpo_carried']

    def run(self):
        for t in range(self.iteration + 1, self.args.iterations + 1):
            self.iteration = t
            result = self.run_iteration()
            self.save_checkpoint(result)
            print(f"iteration {t}: mean reward {np.mean(list(result['mean_reward'].values())):.4f}", flush=True)


def main():
    Trainer(parse_args()).run()


if __name__ == '__main__':
    main()
