import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skillvla.data import H, ChunkDataset, generate_demos  # noqa: E402
from skillvla.env import TASKS  # noqa: E402
from skillvla.flow import sample_tau  # noqa: E402
from skillvla.models import ARM_DIM, build, paired_forward  # noqa: E402
from skillvla.planner import coop_prior, rule_planner  # noqa: E402
from skillvla.text import Tokenizer  # noqa: E402

DEV = "cuda"
METHODS = ["mono", "twin", "skillvla", "skillvla_noattn"]


@pytest.fixture(scope="module")
def data():
    tasks = [TASKS["L_cup"], TASKS["R_stir"], TASKS["ball"]]
    sents = []
    for t in TASKS.values():
        from skillvla.env import BimanualEnv
        g = BimanualEnv(t, 0).reset()["instruction"]
        sents += [g, *rule_planner(g)]
    tok = Tokenizer(sents)
    eps = generate_demos(tasks, 2, seed=0, coop_prior=coop_prior, planner=rule_planner)
    return tok, ChunkDataset(eps, tok, DEV)


def batch(ds, n=16, seed=0):
    return ds.sample(n, torch.Generator(device=DEV).manual_seed(seed))


def randomize_xattn(model):
    for m in model.modules():
        if hasattr(m, "xa"):
            torch.nn.init.normal_(m.xa.out_proj.weight, std=0.2)


def _left_velocity(model, b, x_l, x_r, tau, gate):
    cond = model.conditions(b)
    return paired_forward(model.exp["L"], model.exp["R"], x_l, x_r, tau, *cond, gate)[0]


@torch.no_grad()
def test_gate_off_left_arm_independent_of_right_inputs(data):
    tok, ds = data
    m = build("skillvla", len(tok)).to(DEV).eval()
    randomize_xattn(m)
    b = batch(ds)
    B = b["image"].shape[0]
    tau = sample_tau(B, DEV)
    x_l, x_r = torch.randn(B, H, ARM_DIM, device=DEV), torch.randn(B, H, ARM_DIM, device=DEV)
    b2 = dict(b, u_R=b["u_L"].flip(0), state_R=torch.randn_like(b["state_R"]))
    x_r2 = torch.randn_like(x_r)
    zeros, ones = torch.zeros(B, device=DEV), torch.ones(B, device=DEV)
    v = _left_velocity(m, b, x_l, x_r, tau, zeros)
    v2 = _left_velocity(m, b2, x_l, x_r2, tau, zeros)
    assert torch.equal(v, v2), "gate=0 must remove every right->left path"
    v_on = _left_velocity(m, b, x_l, x_r, tau, ones)
    v_on2 = _left_velocity(m, b2, x_l, x_r2, tau, ones)
    assert (v_on - v_on2).abs().max() > 1e-3, "gate=1 must let right-arm information reach the left arm"


@torch.no_grad()
def test_twin_is_entangled(data):
    tok, ds = data
    m = build("twin", len(tok)).to(DEV).eval()
    b = batch(ds)
    b2 = dict(b, state_R=torch.randn_like(b["state_R"]))
    cl, _, _, _ = m.encode(b)
    cl2, _, _, _ = m.encode(b2)
    assert (cl - cl2).abs().max() > 1e-4, "TwinVLA joint attention should leak right state into left stream"


def test_gate_label_stop_gradient(data):
    """L_disc / L_prior / L_sticky / L_sup train only the estimator; the on/off comparison is stop-grad."""
    tok, ds = data
    m = build("skillvla", len(tok)).to(DEV)
    out = m.loss(batch(ds))
    gate_only = out["loss"] - out["bc"]
    gate_only.backward()
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in m.exp.parameters())
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in m.enc.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in m.coop.parameters())


def test_gate_label_sign(data):
    """y = 1 exactly when communication lowers the BC error (Sec. V-B)."""
    tok, ds = data
    m = build("skillvla", len(tok)).to(DEV)
    out = m.loss(batch(ds))
    # zero-init cross-arm output: on == off  ->  y must be 0 everywhere (strict <, A-023)
    assert float(out["y_frac"]) == 0.0
    assert torch.isclose(out["l_on"], out["l_off"])


@pytest.mark.parametrize("name", METHODS)
def test_overfit_and_sample(data, name):
    tok, ds = data
    torch.manual_seed(0)
    m = build(name, len(tok)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    b = batch(ds, 64)
    first = None
    for _ in range(300):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m.loss(b)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        first = first if first is not None else float(out["bc"])
    assert float(out["bc"]) < 0.5 * first, f"{name}: bc {first:.3f} -> {float(out['bc']):.3f}"
    m.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = m.sample(b)
    assert res[0].shape == (64, H, ARM_DIM) and res[1].shape == (64, H, ARM_DIM)


def test_gate_opens_on_coop_prior(data):
    """With the prior (A-006), the estimator learns gate=1 on coop frames and 0 on single-arm frames."""
    tok, ds = data
    torch.manual_seed(0)
    m = build("skillvla", len(tok)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    for i in range(200):
        b = batch(ds, 64, seed=i)
        out = m.loss(b)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    b = batch(ds, 256, seed=999)
    with torch.no_grad():
        p = m.gate_prob(b["image"], b["instr"])
    coop = b["prior"] > 0.5
    assert (p[coop] >= 0.5).float().mean() > 0.9 and (p[~coop] < 0.5).float().mean() > 0.9


def test_capacity_matched():  # A-016
    count = lambda m, parts: sum(p.numel() for k in parts for p in getattr(m, k).parameters())
    n = {k: count(build(k, 60), ["enc", "exp"]) for k in METHODS}
    ref = n["skillvla"]
    for k, v in n.items():
        assert 0.8 * ref <= v <= 1.2 * ref, (k, v, ref)
