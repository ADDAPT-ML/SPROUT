"""DGPO objective, translated from the authors' code (Luo-Yihong/DGPO, commit 6a45077,
train_sd3_dgpo.py:162-193 and 1132-1156) to a single device.

The functions take tensors only, so they can be tested on CPU.
"""
import torch

UPSTREAM_COMMIT = '6a45077c4b743e6b05ec7b3c2c49bd2be0783694'


def advantages(rewards):
    """Group-standardized advantages, clamped to [-5, 5] (DGPO's GenEval configuration)."""
    std = rewards.std(correction=0)
    return ((rewards - rewards.mean()) / (std + 1e-4)).clamp(-5, 5)


def group_weight(dsm, ref_dsm, advantage, beta, group_size):
    """Detached sigmoid weight of a group, from the advantage-weighted denoising-loss gap to the reference."""
    return torch.sigmoid((advantage * beta * (dsm - ref_dsm) / group_size).sum()).detach()


def sample_surrogate(pred, target, reference, old_dsm, advantage, weight, clip_range, kl_beta):
    """Per-sample loss: clipped policy term against the EMA old policy plus the reference-prediction penalty.

    Returns (loss, clipped, kl)."""
    dsm = (target.float() - pred.float()).square().mean()
    ratio = torch.exp(-dsm.detach() + old_dsm.detach())
    clipped = torch.where(advantage > 0, ratio > 1 + clip_range, ratio < 1 - clip_range)
    policy = weight.detach() * advantage * torch.where(clipped, dsm.detach(), dsm)
    kl = (pred.float() - reference.float()).square().mean()
    return policy + kl_beta * kl, clipped, kl
