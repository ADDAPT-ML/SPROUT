"""Prompt selection: uniform sampling, CGPO, and SPROUT's batch selection (rejection sampling or top-B)."""
import json
import os
import random

import numpy as np
import torch


def uniform(rows, num_prompts):
    """Uniform sampling of training-set rows (the official DGPO sampler). A prompt drawn twice is trained as one
    group of 2G images."""
    return [rows[i] for i in random.sample(range(len(rows)), num_prompts)]


def top_b(prompts, predictions, num_prompts):
    """The B prompts with the highest predicted utility (ties keep pool order)."""
    order = sorted(range(len(prompts)), key=lambda i: predictions[i], reverse=True)
    return [prompts[i] for i in order[:num_prompts]]


def acceptance_probabilities(predictions, clip=0.6, floor=0.1):
    """CGPO's transform of a score into an acceptance probability: clip, min-max scale over the pool, floor."""
    v = np.minimum(np.asarray(predictions, dtype=float), clip)
    span = float(v.max() - v.min())
    probs = np.full_like(v, .5) if span == 0 else (v - v.min()) / span
    return np.maximum(probs, floor)


def rejection_sampling(prompts, predictions, num_prompts, seed, iteration, clip=0.6, floor=0.1):
    """Sweep the pool in random order, accepting each unselected prompt with its acceptance probability, until
    B prompts are accepted (Algorithm 2). Returns (selected, acceptance probabilities)."""
    probs = acceptance_probabilities(predictions, clip, floor)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(iteration), 20261016]))
    selected, chosen = [], set()
    while len(selected) < num_prompts:
        for i in rng.permutation(len(prompts)):
            if prompts[i] in chosen:
                continue
            if rng.random() < probs[i]:
                selected.append(prompts[i])
                chosen.add(prompts[i])
                if len(selected) == num_prompts:
                    break
    return selected, probs


# ---------------------------------------------------------------------------------------------------------------
# CGPO (Li et al., CVPR 2026), ported from the released code in external/cgpo/scripts: the probability update and
# the sampler of train_sd3_cgpo.py (line references L...) and the category calibration of train_sd3_fast_cgpo.py
# (references fast L...). scripts/verify_cgpo_port.py checks the probability update and the sampler against the code.
# ---------------------------------------------------------------------------------------------------------------

CGPO_TASKS = ['single_object', 'two_object', 'colors', 'color_attr', 'position', 'counting']   # fast L655


def cgpo_probabilities(rows, rounds, task_of=None, clip=0.6, floor=0.1):
    """Replay CGPO's per-prompt probability update over completed rounds.

    rows: the training-set rows (CGPO keeps one probability per row; rows of one prompt share it).
    rounds: per completed iteration, {prompt: [rewards of its group]} in training order.
    task_of: prompt -> GenEval task, for CGPO's category calibration (None disables it).
    Returns (sampling probability per prompt, task weight per prompt or None)."""
    prompt_probs = {p: 1.0 for p in rows}                                       # L919-920
    row_count = {}
    for p in rows:
        row_count[p] = row_count.get(p, 0) + 1
    history = {}                                                                # L906
    sampler_probs = dict(prompt_probs)
    task_weight = {t: 1 for t in CGPO_TASKS}                                    # fast L676-682
    for groups in rounds:
        marker = [(p, float(np.var(v, ddof=1))) for p, v in groups.items() if len(v) > 1]
        if not marker:
            continue
        if task_of is not None:
            # fast L876-887: running pairwise average of the mean reward per task, in sorted prompt order.
            task_reward = {t: 0 for t in CGPO_TASKS}
            for q in sorted(groups):
                t, m = task_of[q], float(np.mean(groups[q]))
                task_reward[t] = (m + task_reward[t]) / 2 if task_reward.get(t) != 0 else m
            for t in CGPO_TASKS:                                                # fast L888-919, lambda = 10
                tr = task_reward.get(t)
                if tr > 0:
                    total_inv = sum(1 / task_reward.get(pt, 0) for pt in CGPO_TASKS if task_reward.get(pt, 0) > 0)
                    v = (1 / tr) / total_inv if total_inv > 0 else 1 / len(CGPO_TASKS)
                    task_weight[t] = (1 + 10 * v) * 0.5 + task_weight[t] * 0.5
        variance = [x if x <= clip else clip for _, x in marker]               # L1156
        lo, hi = min(variance), max(variance)                                   # L1158-1160
        scaled = [0.5] * len(variance) if hi == lo else [(x - lo) / (hi - lo) for x in variance]   # L1161-1166
        scaled = [x if x >= floor else floor for x in scaled]                   # L1168
        measured = {p: x for (p, _), x in zip(marker, scaled)}                  # L1176-1178
        explore = 1 / (len(rows) / len(measured))                               # L1191; len(rows) = row count
        new_probs = {q: measured[q] if q in measured else (1 if pr + explore > 1 else pr + explore)
                     for q, pr in prompt_probs.items()}                         # L1182-1195
        prompt_probs = new_probs                                                # L1202
        updated = {}
        for q, pr in new_probs.items():                                         # L1204-1215
            h = history.get(q)
            if h:
                avg = (pr + sum(h)) / (len(h) + 1)
                updated[q] = floor if avg < floor else (1 if avg > 1 else avg)
            else:
                updated[q] = pr
        for q, pr in new_probs.items():                                         # L1217-1225: once per row
            h = history.setdefault(q, [])
            for _ in range(row_count[q]):
                h.append(pr)
                if len(h) > 2:
                    h.pop(0)
        sampler_probs = updated                                                 # L1226
    weights = {q: task_weight[task_of[q]] for q in prompt_probs} if task_of is not None else None
    return sampler_probs, weights


def cgpo(rows, num_prompts, rounds, carried, seed, iteration, task_of=None):
    """CGPO's sampler (DistributedKRepeatSampler, L274-324): draw B rows at random, accept each with its
    probability, and repeat until B are accepted; accepted rows beyond B carry over to the next iteration.
    Returns (selected, carried over)."""
    probs, weights = cgpo_probabilities(rows, rounds, task_of)
    container = [q for q in carried if q in probs]
    g = torch.Generator()
    g.manual_seed(seed + iteration)                                             # L281-282: seed + epoch
    guard = 0
    while len(container) < num_prompts and guard < 100000:
        guard += 1
        for i in torch.randperm(len(rows), generator=g)[:num_prompts].tolist():  # L286
            draw = torch.rand(1, generator=g).item()
            if draw < probs[rows[i]] * (weights[rows[i]] if weights else 1):    # L290-294; weight: fast L117
                container.append(rows[i])
    return container[:num_prompts], container[num_prompts:]                     # L297-302


def load_rounds(run_dir, iteration):
    """Rewards of iterations 1..iteration-1, read back from the run directory (so CGPO's state survives resume)."""
    rounds = []
    for t in range(1, iteration):
        groups = {}
        with open(os.path.join(run_dir, f'rollouts_{t}.jsonl')) as f:
            for row in map(json.loads, f):
                groups.setdefault(row['prompt'], []).append(float(row['reward']))
        rounds.append(groups)
    return rounds
