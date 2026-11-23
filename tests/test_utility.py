import numpy as np
import pytest

from sprout import utility


def records():
    return [dict(prompt=p, label=float(i), iteration=t) for i, (p, t) in enumerate(
        [('a', 1), ('b', 1), ('a', 2), ('c', 2), ('b', 3)])]


def test_label_histories():
    r = records()
    assert [x['label'] for x in utility.label_history(r, 3, 'all')] == [0, 1, 2, 3, 4]
    assert sorted(x['label'] for x in utility.label_history(r, 3, 'latest')) == [2, 3, 4]
    assert [x['label'] for x in utility.label_history(r, 3, 'previous')] == [4]
    with pytest.raises(ValueError):
        utility.label_history(r, 2, 'all')        # a label from iteration 3 cannot enter a fit after iteration 2


def test_update_norm_labels_are_log_ratios_to_the_median_with_a_floor():
    labels = utility.update_norm_labels({'a': 1.0, 'b': 2.0, 'c': 4.0, 'd': 0.0})
    assert labels['a'] == pytest.approx(np.log(1 / 1.5))
    assert labels['d'] == pytest.approx(np.log(utility.UPDATE_NORM_FLOOR))


def test_variance_label_is_unbiased():
    assert utility.variance_label([0.0, 1.0]) == 0.5


def test_ridge_fit_and_predict(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    emb = {f'p{i}': rng.normal(size=16).astype(np.float32) for i in range(40)}
    monkeypatch.setattr(utility, 'embed', lambda prompts, batch_size=256: np.stack([emb[p] for p in prompts]))
    utility._EMBEDDING_MEMO.clear()
    w = rng.normal(size=16)
    prompts = list(emb)
    y = [float(emb[p] @ w) + 1.0 for p in prompts]
    model = utility.RidgeUtility.fit(prompts, y, alpha=1e-3)
    assert np.allclose([model.predict(p) for p in prompts], y, atol=1e-2)
    model.save(tmp_path / 'm.json')
    loaded = utility.load_utility(tmp_path / 'm.json')
    assert [loaded.predict(p) for p in prompts] == [model.predict(p) for p in prompts]
    utility._EMBEDDING_MEMO.clear()
