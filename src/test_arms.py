"""Invariant checks for the intervention arms. Run: python test_arms.py"""

from __future__ import annotations

import torch

import steering as S
from common import DATA, DEVICE, HOOK, ActStats, load_model, load_sae
from denoiser import WienerDenoiser, build_denoiser, transported_direction

TOL = 1e-4


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    assert cond, name


def main() -> None:
    stats = ActStats.load()
    sae = load_sae()
    model = load_model()
    scale = stats.median_norm
    f = 0
    v = sae.W_dec[f].detach().float()
    v_hat = v / v.norm()

    toks = model.to_tokens(["The tower stands above the river and the"])
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=HOOK)
    h = cache[HOOK].detach()

    wiener = WienerDenoiser(stats, sigma=1.0 * scale, shrink=0.05)
    mlp = build_denoiser("mlp", stats, hidden=256, scale=scale, cond=True).to(DEVICE)

    for label, d in (("wiener", wiener), ("mlp-untrained", mlp)):
        cds = S.DenoiserArm(denoiser=d, mode="cds", lam=1.0)
        out0 = cds(h, v_hat, 0.0)
        check(f"cds identity at s=0 ({label})", torch.allclose(out0, h, atol=TOL))
        s = 1.5 * scale
        out = cds(h, v_hat, s)
        coord_naive = ((h + s * v_hat) @ v_hat)
        check(
            f"cds preserves concept coordinate ({label})",
            torch.allclose(out @ v_hat, coord_naive, atol=1e-2),
            f"max diff {(out @ v_hat - coord_naive).abs().max().item():.5f}",
        )

    naive_arm = S.DenoiserArm(denoiser=wiener, mode="naive", eta=1.0)
    check(
        "task's literal arm does move the activation at s=0",
        not torch.allclose(naive_arm(h, v_hat, 0.0), h, atol=TOL),
        "-- this is the defect the contrastive form removes",
    )

    d_tr = transported_direction(stats, v_hat, 0.05)
    check(
        "mts unit increment along v",
        abs(float(torch.dot(d_tr, v_hat)) - 1.0) < 1e-3,
        f"got {float(torch.dot(d_tr, v_hat)):.6f}",
    )
    mts = S.TransportedArm(direction=d_tr)
    s = 2.0 * scale
    check(
        "mts matches naive concept coordinate",
        torch.allclose(mts(h, v_hat, s) @ v_hat, (h + s * v_hat) @ v_hat, atol=1e-2),
    )
    check(
        "mts is not proportional to v",
        float(torch.nn.functional.cosine_similarity(d_tr, v_hat, dim=0)) < 0.999,
        f"cos={float(torch.nn.functional.cosine_similarity(d_tr, v_hat, dim=0)):.4f}",
    )

    blob = torch.load(DATA / "sae_feature_stats.pt", map_location=DEVICE)
    ceiling = blob["max_act"].to(DEVICE)
    w = sae.W_dec / sae.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    exempt = (w @ v_hat).abs() > 0.3
    exempt[f] = True
    fsr = S.FeatureSurgeryArm(sae=sae, ceiling=ceiling, exempt=exempt, k=1.0)
    with torch.no_grad():
        check("fsr identity at s=0", torch.allclose(fsr(h, v_hat, 0.0), h, atol=TOL))
        out = fsr(h, v_hat, 3.0 * scale)
        check("fsr changes the activation under steering", not torch.allclose(out, h, atol=TOL))
        rep = fsr.clamp_report(h, v_hat, 3.0 * scale)
        check("fsr clamps something at high strength", rep["n_over_per_token"] > 0, str(rep["n_over_per_token"]))

    state = S.HookState()
    for name, fn in (("naive", S.naive), ("mts", mts), ("fsr", fsr)):
        state.reset()
        with model.hooks(fwd_hooks=[(HOOK, S.make_hook(fn, v_hat, 1.0 * scale, state))]):
            with torch.no_grad():
                gen = model.generate(
                    toks, max_new_tokens=6, do_sample=False, stop_at_eos=False, verbose=False
                )
        check(f"generation runs with {name} hook", gen.shape[1] == toks.shape[1] + 6)
        print(f"       -> {model.to_string(gen[0, toks.shape[1]:])!r}")

    state.reset()
    with model.hooks(fwd_hooks=[(HOOK, S.make_hook(S.naive, v_hat, 0.0, state))]):
        with torch.no_grad():
            a = model.generate(toks, max_new_tokens=6, do_sample=False, stop_at_eos=False, verbose=False)
    with torch.no_grad():
        b = model.generate(toks, max_new_tokens=6, do_sample=False, stop_at_eos=False, verbose=False)
    check("zero-strength hook is a no-op end to end", torch.equal(a, b))
    print("all invariants hold")


if __name__ == "__main__":
    main()


def test_direction_cache_cannot_collide():
    """The cache must not be able to serve one feature another feature's correction.

    The original defect keyed the cache on `id(v_hat)` while storing only the result. `generate.py` binds a
    fresh short-lived direction per feature, so CPython recycled the address and a later feature silently
    got an earlier one's correction -- every number stayed finite and plausible.

    Address reuse cannot be forced portably, so this checks the two things that make the collision
    impossible rather than waiting for the allocator to cooperate: the cache keeps a reference to the
    direction it was computed from (which is what stops the address from being recycled at all), and a
    lookup that lands on a stale key recomputes instead of returning the wrong direction. The second is
    tested by planting a collision by hand -- exactly what the allocator used to do by accident.
    """
    import torch

    from steering import CorrectedDirectionArm

    class Tag(torch.nn.Module):
        """Returns a direction traceable to the input it was built from."""

        def forward(self, x):
            return x * 3.0

    arm = CorrectedDirectionArm(correction=Tag())

    a = torch.tensor([1.0, 0.0, 0.0])
    w_a = arm._w(a)
    assert torch.allclose(w_a, a * 3.0)

    # The mechanism: the direction itself is retained, so its address cannot be handed to a later feature.
    assert any(entry[0] is a for entry in arm.cache.values()), (
        "the cache does not keep the direction alive, so its address can be recycled and a later feature "
        "will hit this entry"
    )

    # The guard: plant the collision the allocator used to produce, and check it is caught.
    b = torch.tensor([0.0, 2.0, 0.0])
    arm.cache[id(b)] = (a, w_a)
    assert torch.allclose(arm._w(b), b * 3.0), (
        "a stale cache key returned another direction's correction instead of recomputing"
    )
    print("[ok ] the direction cache cannot serve one feature another feature's correction")
