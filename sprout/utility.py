"""Utility labels, label histories and text-only utility models."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

UPDATE_NORM_FLOOR = 0.0199


# ---------------------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------------------

def variance_label(rewards):
    """Unbiased variance of a prompt's group rewards (SPROUT w/ variance)."""
    return float(np.asarray(rewards, dtype=float).var(ddof=1))


def update_norm_labels(norms):
    """SPROUT's label: log of each prompt's update norm over the median of the iteration. The floor avoids
    log(0) for a group with constant rewards at iteration 1, where the model equals the reference."""
    median = np.median(list(norms.values()))
    return {p: float(np.log(max(n / median, UPDATE_NORM_FLOOR))) for p, n in norms.items()}


def label_history(records, iteration, mode):
    """Records used to fit after iteration `iteration`: all, the latest per prompt, or the previous iteration."""
    if any(int(r['iteration']) > iteration for r in records):
        raise ValueError('a label from a later iteration cannot enter the fit')
    if mode == 'all':
        return list(records)
    if mode == 'latest':
        latest = {}
        for r in records:
            if r['prompt'] not in latest or r['iteration'] >= latest[r['prompt']]['iteration']:
                latest[r['prompt']] = r
        return list(latest.values())
    if mode == 'previous':
        return [r for r in records if int(r['iteration']) == iteration]
    raise ValueError(f'unknown label history {mode!r}')


# ---------------------------------------------------------------------------------------------------------------
# Ridge regression on frozen CLIP ViT-L/14 text embeddings
# ---------------------------------------------------------------------------------------------------------------

_CLIP = {}
_EMBEDDING_MEMO = {}


def clip_text_model():
    if 'model' not in _CLIP:
        import clip
        model, _ = clip.load('ViT-L/14', device='cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        _CLIP['model'] = model
    return _CLIP['model']


def embed(prompts, batch_size=256):
    """L2-normalized CLIP text embeddings, shape (n, 768)."""
    import clip
    model = clip_text_model()
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            e = model.encode_text(clip.tokenize(list(prompts)[i:i + batch_size], truncate=True).to(device)).float()
            out.append((e / e.norm(dim=-1, keepdim=True).clamp(min=1e-8)).cpu().numpy())
    return np.concatenate(out, axis=0)


class RidgeUtility:
    """Ridge regression on standardized embeddings, with an unpenalized intercept."""

    def __init__(self, mu, sigma, weights, bias):
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = np.array(sigma, dtype=np.float32)
        self.weights = np.array(weights, dtype=np.float32)
        self.bias = float(bias)

    @classmethod
    def fit(cls, prompts, labels, alpha):
        X = embed(prompts)
        y = np.array(labels, dtype=np.float32)
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        sigma[sigma < 1e-6] = 1.0
        Xa = np.concatenate([(X - mu) / sigma, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
        penalty = np.eye(Xa.shape[1], dtype=np.float32)
        penalty[-1, -1] = 0.0
        w = np.linalg.solve(Xa.T @ Xa + alpha * penalty, Xa.T @ y)
        return cls(mu, sigma, w[:-1], float(w[-1]))

    def predict(self, prompt):
        # Prompts are embedded one at a time at prediction (memoized; the encoder is frozen).
        x = _EMBEDDING_MEMO.get(prompt)
        if x is None:
            x = _EMBEDDING_MEMO[prompt] = embed([prompt])[0]
        return float(np.dot((x - self.mu) / self.sigma, self.weights) + self.bias)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(dict(type='ridge', mu=self.mu.tolist(), sigma=self.sigma.tolist(),
                           weights=self.weights.tolist(), bias=self.bias), f)


# ---------------------------------------------------------------------------------------------------------------
# Fine-tuned CLIP text tower (predictor ablation, B=16)
# ---------------------------------------------------------------------------------------------------------------

MINIBATCH = 32
_LIVE = {}      # resolved weights path -> (model in eval mode, label mean, label SD)


class TextRegressor(torch.nn.Module):
    """CLIP ViT-L/14 text tower with a 768-256-1 head on its L2-normalized output."""

    def __init__(self, clip_model, head_seed):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        torch.manual_seed(head_seed)
        self.head = torch.nn.Sequential(torch.nn.Linear(768, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.positional_embedding
        x = self.ln_final(self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2))
        e = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
        return self.head(e / e.norm(dim=-1, keepdim=True)).squeeze(1)


def _build_regressor(head_seed, device):
    """Building the model does not advance the global random state."""
    import clip
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        base, _ = clip.load('ViT-L/14', device='cpu', jit=False)
        model = TextRegressor(base.float(), head_seed)
    del base
    return model.to(device)


def _tokens(prompts, device):
    import clip
    return clip.tokenize(list(prompts), truncate=True).to(device)


class FinetunedCLIPUtility:
    """Warm-started across iterations: each refit is one shuffled pass (minibatch 32) over all labels so far,
    on labels standardized with their mean and SD; predictions are mapped back to label units."""

    def __init__(self, seed, backbone_lr, head_lr, device='cuda'):
        self.seed, self.device = int(seed), device
        self.model = _build_regressor(self.seed, device)
        self.opt = torch.optim.Adam([
            {'params': self.model.head.parameters(), 'lr': float(head_lr)},
            {'params': [p for n, p in self.model.named_parameters() if not n.startswith('head.')],
             'lr': float(backbone_lr)}])
        self.label_mu, self.label_sd = 0.0, 1.0

    def update(self, prompts, labels, iteration):
        y = np.asarray(labels, dtype=float)
        self.label_mu, self.label_sd = float(y.mean()), float(y.std() + 1e-8)
        z = (y - self.label_mu) / self.label_sd
        order = np.random.default_rng(np.random.SeedSequence([self.seed, int(iteration), 20261027])).permutation(len(y))
        self.model.train()
        for k in range(0, len(order), MINIBATCH):
            idx = order[k:k + MINIBATCH]
            self.opt.zero_grad(set_to_none=True)
            out = self.model(_tokens([prompts[i] for i in idx], self.device))
            loss = ((out - torch.tensor(z[idx], dtype=torch.float32, device=self.device)) ** 2).mean()
            loss.backward()
            self.opt.step()
        self.model.eval()

    @staticmethod
    def names(iteration):
        return f'clip_ft_iter_{iteration}.pt', f'clip_ft_optim_iter_{iteration}.pt'

    def save(self, models_dir, iteration):
        """Write the weights and the Adam state of `iteration`; keep the files of the three latest iterations."""
        models_dir = Path(models_dir)
        weights, optim = self.names(iteration)
        for name, value in ((weights, dict(model=self.model.state_dict(), label_mu=self.label_mu,
                                           label_sd=self.label_sd, seed=self.seed, iteration=int(iteration))),
                            (optim, dict(optimizer=self.opt.state_dict(), iteration=int(iteration)))):
            tmp = models_dir / f'{name}.tmp.{os.getpid()}'
            torch.save(value, tmp)
            os.replace(tmp, models_dir / name)
        _LIVE.clear()
        _LIVE[str((models_dir / weights).resolve())] = (copy.deepcopy(self.model).eval(), self.label_mu, self.label_sd)
        for f in models_dir.glob('clip_ft_*iter_*.pt'):
            if int(f.stem.rsplit('_', 1)[1]) < int(iteration) - 2:
                f.unlink()
        return weights, optim

    def load(self, models_dir, iteration):
        weights, optim = self.names(iteration)
        w = torch.load(Path(models_dir) / weights, map_location=self.device)
        o = torch.load(Path(models_dir) / optim, map_location=self.device)
        if w['seed'] != self.seed or w['iteration'] != iteration or o['iteration'] != iteration:
            raise RuntimeError('fine-tuned CLIP state does not match this run')
        self.model.load_state_dict(w['model'], strict=True)
        self.opt.load_state_dict(o['optimizer'])
        self.label_mu, self.label_sd = float(w['label_mu']), float(w['label_sd'])
        self.model.eval()


class FinetunedCLIPPredictor:
    def __init__(self, weights_path):
        self.weights_path = weights_path

    @torch.no_grad()
    def predict(self, prompt, device='cuda'):
        key = str(Path(self.weights_path).resolve())
        if key not in _LIVE:
            w = torch.load(key, map_location=device)
            model = _build_regressor(int(w['seed']), device)
            model.load_state_dict(w['model'], strict=True)
            _LIVE.clear()
            _LIVE[key] = (model.eval(), float(w['label_mu']), float(w['label_sd']))
        model, mu, sd = _LIVE[key]
        return float((np.asarray([float(model(_tokens([prompt], device))[0])]) * sd + mu)[0])


def load_utility(path):
    """The utility model committed at `path` (written after an iteration, used to select the next batch)."""
    with open(path) as f:
        spec = json.load(f)
    if spec['type'] == 'ridge':
        return RidgeUtility(spec['mu'], spec['sigma'], spec['weights'], spec['bias'])
    return FinetunedCLIPPredictor(os.path.join(os.path.dirname(os.path.abspath(path)), spec['weights_file']))
