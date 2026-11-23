"""Training prompt pool."""
import random

from sprout.reward import load_metadata_rows, normalize_prompt


def load_pool(metadata_path):
    """Prompts of the training set (DGPO's GenEval train_metadata.jsonl, 50,000 rows) in file order. The official
    sampler draws rows, so a prompt with r rows is r times as likely under uniform sampling.

    The shuffle of the distinct prompts does not change the returned rows. It is kept because it advances the global
    `random` state, on which the prompt draws of the paper's runs depend."""
    rows = [normalize_prompt(r['prompt']) for r in load_metadata_rows(metadata_path)]
    random.shuffle(sorted(set(rows)))
    return rows


def distinct(rows):
    """Distinct prompts in order of first appearance."""
    return list(dict.fromkeys(rows))
