"""The per-sample DGPO surrogate, accumulated sample by sample, gives the gradient of the full-batch objective."""
import torch

from sprout.dgpo_loss import advantages, group_weight, sample_surrogate


def test_streaming_matches_full_graph_with_clipping_and_kl():
    torch.manual_seed(197)
    for shift in (0., 2., -2.):
        pred = torch.randn(2, 24, 8, requires_grad=True)
        target, ref = torch.randn_like(pred), torch.randn_like(pred)
        adv = torch.stack([advantages(r) for r in torch.randn(2, 24)])
        dsm = (target - pred).square().mean(-1)
        ref_dsm = (target - ref).square().mean(-1)
        old_dsm = dsm.detach() + shift
        weights = torch.sigmoid((adv * 100 * (dsm.detach() - ref_dsm) / 24).sum(-1))
        ratio = torch.exp(-dsm.detach() + old_dsm)
        mask = torch.where(adv > 0, ratio > 1.01, ratio < .99)
        policy = weights[:, None] * adv * torch.where(mask, dsm.detach(), dsm)
        expected, = torch.autograd.grad((policy + .02 * (pred - ref).square().mean(-1)).mean(), pred)
        actual = torch.zeros_like(pred)
        for g in range(2):
            w = group_weight(dsm[g].detach(), ref_dsm[g], adv[g], 100, 24)
            for i in range(24):
                x = pred[g, i].detach().clone().requires_grad_()
                value, clipped, _ = sample_surrogate(x, target[g, i], ref[g, i], old_dsm[g, i], adv[g, i], w, 0.01, 0.02)
                actual[g, i], = torch.autograd.grad(value / 48, x)
                assert bool(clipped) == bool(mask[g, i])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_constant_rewards_keep_only_the_reference_penalty():
    a = advantages(torch.ones(24))
    assert float(a.abs().sum()) == 0.
    w = group_weight(torch.ones(24), torch.zeros(24), a, 100, 24)
    assert float(w) == .5
    x = torch.ones(3, requires_grad=True)
    loss, _, _ = sample_surrogate(x, torch.zeros(3), torch.zeros(3), torch.tensor(1.), a[0], w, 0.01, 0.02)
    grad, = torch.autograd.grad(loss, x)
    torch.testing.assert_close(grad, torch.full((3,), .04 / 3))


def test_unclipped_gradient_is_a_scaled_softplus():
    torch.manual_seed(211)
    dsm = torch.rand(24, requires_grad=True)
    ref = dsm.detach() + torch.randn(24) * .01
    a = advantages(torch.randn(24))
    w = group_weight(dsm.detach(), ref, a, 100, 24)
    g, = torch.autograd.grad((w * a * dsm).mean(), dsm)
    h, = torch.autograd.grad(torch.nn.functional.softplus((a * 100 * (dsm - ref) / 24).sum()) / 100, dsm)
    torch.testing.assert_close(g, h)
