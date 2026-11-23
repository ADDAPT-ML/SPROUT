from sprout.train import lr_at, parse_args


def schedule(*extra):
    return vars(parse_args(['--output_dir', 'x', '--seed', '0', '--selection', 'uniform', *extra]))


def test_constant():
    a = schedule()
    assert {lr_at(a, t) for t in range(1, 101)} == {1.65e-4}


def test_late_decay_with_tail():
    # the 200-iteration runs: constant to 100, cosine to 0.3 lr by 120, then a second cosine to 0 at 200
    a = schedule('--iterations', '200', '--lr_schedule', 'cosine', '--lr_decay_start', '100', '--lr_decay_iters', '20',
                 '--lr_min_ratio', '0.3', '--lr_tail_ratio', '0.0')
    assert lr_at(a, 100) == 1.65e-4
    assert abs(lr_at(a, 120) - 0.3 * 1.65e-4) < 1e-12
    assert lr_at(a, 200) < 1e-12
    assert all(lr_at(a, t + 1) <= lr_at(a, t) for t in range(1, 200))


def test_ema_calls_follow_the_batch():
    assert schedule()['ema_calls_per_update'] == 3
    assert schedule('--num_prompts', '16')['ema_calls_per_update'] == 6
