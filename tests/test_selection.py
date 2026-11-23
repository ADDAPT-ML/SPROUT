import random

import numpy as np

from sprout import selection


def test_rejection_sampling_returns_distinct_prompts_and_is_seeded():
    prompts = [f'p{i}' for i in range(500)]
    pred = np.random.default_rng(0).normal(size=500)
    a, probs = selection.rejection_sampling(prompts, pred, 8, seed=1, iteration=3)
    b, _ = selection.rejection_sampling(prompts, pred, 8, seed=1, iteration=3)
    c, _ = selection.rejection_sampling(prompts, pred, 8, seed=1, iteration=4)
    assert a == b != c and len(set(a)) == 8
    assert probs.min() == 0.1 and probs.max() == 1.0


def test_acceptance_probabilities_clip_scale_and_floor():
    probs = selection.acceptance_probabilities([0.0, 0.3, 0.6, 2.0], clip=0.6, floor=0.1)
    np.testing.assert_allclose(probs, [0.1, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(selection.acceptance_probabilities([1.0, 1.0]), [0.5, 0.5])


def test_rejection_sampling_favors_high_predictions():
    prompts = [f'p{i}' for i in range(100)]
    pred = np.r_[np.zeros(50), np.ones(50)]
    counts = np.zeros(2)
    for it in range(200):
        sel, _ = selection.rejection_sampling(prompts, pred, 8, 0, it)
        for p in sel:
            counts[int(int(p[1:]) >= 50)] += 1
    assert counts[1] > 5 * counts[0]


def test_top_b_keeps_pool_order_on_ties():
    assert selection.top_b(['a', 'b', 'c', 'd'], [1, 3, 3, 0], 2) == ['b', 'c']


def test_uniform_draws_rows():
    random.seed(0)
    rows = ['x', 'x', 'y']
    assert all(p in rows for p in selection.uniform(rows, 2))


def test_cgpo_probabilities_start_at_one_and_demote_low_variance_prompts():
    rows = ['a', 'b', 'c', 'd']
    probs, weights = selection.cgpo_probabilities(rows, [])
    assert probs == {p: 1.0 for p in rows} and weights is None
    rounds = [{'a': [0, 1, 0, 1], 'b': [0.5, 0.5, 0.5, 0.6]}]
    probs, _ = selection.cgpo_probabilities(rows, rounds)
    assert probs['a'] == 1.0 and probs['b'] == 0.1      # min-max over the measured batch, floor 0.1
    assert probs['c'] == probs['d'] == 1.0              # unmeasured prompts stay at the cap


def test_cgpo_carries_overflow():
    rows = [f'p{i}' for i in range(50)]
    sel, carried = selection.cgpo(rows, 8, [], [], seed=197, iteration=1)
    assert len(sel) == 8
    sel2, _ = selection.cgpo(rows, 8, [], carried + ['not-in-pool'], seed=197, iteration=2)
    assert sel2[:len(carried)] == carried[:8]
