"""Check the CGPO port (sprout/selection.py) against CGPO's released code, executed verbatim.

1. Probabilities: lines 1145-1227 of external/cgpo/scripts/train_sd3_cgpo.py (the probability update), fed with
   the per-prompt variances of CGPO's own PerPromptStatTracker.curriculum_filter, round by round over the rewards
   of a training run; compared with selection.cgpo_probabilities after every round. Passes if identical.
2. Sampler: CGPO's DistributedKRepeatSampler draws two batches from a probability table with demotions; compared
   with selection.cgpo, including the rows carried over to the second batch. CGPO then shuffles each batch to split
   it across GPUs; on one device the accepted order is kept, so the batches are compared as multisets.

usage: python scripts/verify_cgpo_port.py <run_dir> [<run_dir> ...]
"""
import importlib.util
import os
import sys
import textwrap

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CGPO = os.path.join(REPO, 'external', 'cgpo')
sys.path.insert(0, REPO)
from sprout import selection  # noqa: E402

spec = importlib.util.spec_from_file_location('stc', os.path.join(CGPO, 'flow_grpo', 'stat_tracking_cur.py'))
stc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stc)
SOURCE = open(os.path.join(CGPO, 'scripts', 'train_sd3_cgpo.py')).read().split('\n')
UPDATE = textwrap.dedent('\n'.join(SOURCE[1144:1227]))
assert UPDATE.lstrip().startswith('prompt_list = []')
assert 'train_sampler.update_probs(updated_prompt_probs)' in UPDATE.splitlines()[-1]
UPDATE = compile(UPDATE, 'train_sd3_cgpo.py:1145-1227', 'exec')


class _Sampler:
    def update_probs(self, probs):
        self.prompt_probs = probs


def cgpo_reference(pool, rounds):
    """CGPO's probability table after each round."""
    ns = {'prompt_probs': [(p, 1) for p in pool], 'prompt_history': {}, 'train_sampler': _Sampler()}
    out = []
    for groups in rounds:
        tracker = stc.PerPromptStatTracker()
        ns['selection_marker'] = [(p, tracker.curriculum_filter(np.asarray(v, dtype=np.float64).reshape(-1, 1)))
                                  for p, v in groups.items()]
        exec(UPDATE, ns)
        out.append(dict(ns['train_sampler'].prompt_probs))
    return out


def check_probabilities(pool, rounds):
    reference = cgpo_reference(pool, rounds)
    worst = 0.0
    for k in range(1, len(rounds) + 1):
        ours, _ = selection.cgpo_probabilities(pool, rounds[:k])
        worst = max(worst, max(abs(ours[p] - reference[k - 1][p]) for p in pool))
    print(f'probabilities over {len(rounds)} rounds: max |ours - CGPO| = {worst:.3e}')
    return worst == 0.0


def check_sampler(pool, rounds, seed=197, batch=20, epoch=4):
    from torch.utils.data import Sampler
    a = next(i for i, line in enumerate(SOURCE) if line.startswith('class DistributedKRepeatSampler'))
    b = next(i for i in range(a + 1, len(SOURCE)) if SOURCE[i] and not SOURCE[i][0].isspace()
             and not SOURCE[i].startswith('#'))
    ns = {'torch': torch, 'Sampler': Sampler}
    exec(compile('\n'.join(SOURCE[a:b]), 'train_sd3_cgpo.py:DistributedKRepeatSampler', 'exec'), ns)
    probs = cgpo_reference(pool, rounds[:3])[-1]
    sampler = ns['DistributedKRepeatSampler'](pool, batch_size=batch, k=1, num_replicas=1, rank=0, seed=seed)
    sampler.update_probs([(p, probs[p]) for p in pool])
    it = iter(sampler)
    sampler.set_epoch(epoch)
    first = [pool[i] for i in next(it)]
    sampler.set_epoch(epoch + 1)
    second = [pool[i] for i in next(it)]
    ours1, carried = selection.cgpo(pool, batch, rounds[:3], [], seed, epoch)
    ours2, _ = selection.cgpo(pool, batch, rounds[:3], carried, seed, epoch + 1)
    same = sorted(first) == sorted(ours1) and sorted(second) == sorted(ours2)
    print(f'sampler: two batches contain the same rows as CGPO\'s: {same}')
    return same


def main():
    ok = True
    for run in sys.argv[1:]:
        rounds = selection.load_rounds(run, 1 + sum(1 for f in os.listdir(run) if f.startswith('rollouts_')))
        pool = sorted({p for g in rounds for p in g} | {f'__unsampled_{i}' for i in range(3000)})
        ok &= check_probabilities(pool, rounds) and check_sampler(pool, rounds)
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
