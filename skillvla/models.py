"""Policies: Mono (pi0.5-like), TwinVLA-like, SkillVLA, SkillVLA w/o Attn.

All share the same building blocks: a CNN image tokenizer, a word-embedding
text tokenizer, a transformer prefix encoder (our stand-in for the VLM,
A-001/A-022), and a transformer flow-matching action expert over an H-step
action chunk.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import H
from .flow import flow_loss, flow_sample, noisy, sample_tau
from .text import MAX_LEN

STATE_DIM, ARM_DIM = 7, 4


# ---------------------------------------------------------------- blocks
class ImageTokens(nn.Module):
    """64x64 RGB -> 8x8 = 64 tokens."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, d, 3, 2, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        self.pos = nn.Parameter(torch.randn(64, d) * 0.02)

    def forward(self, img):
        x = img.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return self.net(x).flatten(2).transpose(1, 2) + self.pos


class Block(nn.Module):
    """Pre-LN transformer block: self-attn [+ cross-attn] + MLP."""

    def __init__(self, d, heads=4, cross=False):
        super().__init__()
        self.n1, self.sa = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.nc, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attend(self, x, kv=None, kv_pad=None):
        """Self-attn over [x ; kv] (kv tokens are extra keys, used for TwinVLA joint attention)."""
        h = self.n1(x)
        k = h if kv is None else torch.cat([h, self.n1(kv)], 1)
        return x + self.sa(h, k, k, key_padding_mask=kv_pad, need_weights=False)[0]

    def forward(self, x, pad=None, ctx=None, ctx_pad=None):
        x = self.attend(x, kv_pad=pad)
        if self.cross:
            c = self.nkv(ctx)
            x = x + self.ca(self.nc(x), c, c, key_padding_mask=ctx_pad, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class PrefixEncoder(nn.Module):
    """(image, text, state tokens) -> prefix tokens. Stand-in for a (low-level) VLM."""

    def __init__(self, vocab, d, layers, n_state):
        super().__init__()
        self.img = ImageTokens(d)
        self.tok = nn.Embedding(vocab, d)
        self.tpos = nn.Parameter(torch.randn(MAX_LEN, d) * 0.02)
        self.state = nn.ModuleList(nn.Linear(STATE_DIM, d) for _ in range(n_state))
        self.kind = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d) for _ in range(layers))
        self.out = nn.LayerNorm(d)

    def embed(self, img, ids, states):
        t = [self.img(img) + self.kind[0], self.tok(ids) + self.tpos + self.kind[1]]
        t += [lin(s)[:, None] + self.kind[2] for lin, s in zip(self.state, states)]
        x = torch.cat(t, 1)
        pad = torch.cat([torch.zeros(img.shape[0], 64, dtype=torch.bool, device=img.device), ids == 0,
                         torch.zeros(img.shape[0], len(states), dtype=torch.bool, device=img.device)], 1)
        return x, pad

    def forward(self, img, ids, states):
        x, pad = self.embed(img, ids, states)
        for b in self.blocks:
            x = b.attend(x, kv_pad=pad)
            x = x + b.mlp(b.n2(x))
        return self.out(x), pad


def time_embedding(tau, d):
    half = d // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device) / half)
    a = tau[:, None].float() * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ExpertLayer(nn.Module):
    """Action-expert layer: self-attn over chunk, cross-attn to prefix, optional cross-arm attn, MLP."""

    def __init__(self, d, heads=4, arm_xattn=False):
        super().__init__()
        self.base = Block(d, heads, cross=True)
        self.arm_xattn = arm_xattn
        if arm_xattn:  # independent QKV trained from scratch; zero-init output (A-021)
            self.nx, self.nm = nn.LayerNorm(d), nn.LayerNorm(d)
            self.xa = nn.MultiheadAttention(d, heads, batch_first=True)
            nn.init.zeros_(self.xa.out_proj.weight)
            nn.init.zeros_(self.xa.out_proj.bias)

    def pre(self, x, ctx, ctx_pad):
        b = self.base
        x = b.attend(x)
        c = b.nkv(ctx)
        return x + b.ca(b.nc(x), c, c, key_padding_mask=ctx_pad, need_weights=False)[0]

    def message(self, x, other):
        o = self.nm(other)
        return self.xa(self.nx(x), o, o, need_weights=False)[0]

    def post(self, x):
        return x + self.base.mlp(self.base.n2(x))


class Expert(nn.Module):
    def __init__(self, d, layers, act_dim, arm_xattn=False):
        super().__init__()
        self.inp = nn.Linear(act_dim, d)
        self.pos = nn.Parameter(torch.randn(H, d) * 0.02)
        self.temb = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(ExpertLayer(d, arm_xattn=arm_xattn) for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, act_dim))
        self.d = d

    def embed(self, x, tau):
        return self.inp(x) + self.pos + self.temb(time_embedding(tau, self.d))[:, None]

    def forward(self, x, tau, ctx, ctx_pad):
        h = self.embed(x, tau)
        for layer in self.layers:
            h = layer.post(layer.pre(h, ctx, ctx_pad))
        return self.out(h)


def paired_forward(exp_l, exp_r, x_l, x_r, tau, ctx_l, pad_l, ctx_r, pad_r, gate):
    """Two action experts with (gated) cross-arm attention.

    gate: (B,) in [0,1]; messages at a layer are computed from both arms' pre-update states.
    gate == 0 removes every cross-arm path exactly (no leakage through the message branch).
    """
    h_l, h_r = exp_l.embed(x_l, tau), exp_r.embed(x_r, tau)
    g = gate.view(-1, 1, 1).to(h_l.dtype)
    for ll, lr in zip(exp_l.layers, exp_r.layers):
        h_l, h_r = ll.pre(h_l, ctx_l, pad_l), lr.pre(h_r, ctx_r, pad_r)
        if ll.arm_xattn:
            m_l, m_r = ll.message(h_l, h_r), lr.message(h_r, h_l)
            h_l, h_r = h_l + g * m_l, h_r + g * m_r
        h_l, h_r = ll.post(h_l), lr.post(h_r)
    return exp_l.out(h_l), exp_r.out(h_r)


# ---------------------------------------------------------------- policies
class Mono(nn.Module):
    """pi0.5-like: one shared prefix (image, global instruction, both states) -> one expert on 8-D joint actions."""

    def __init__(self, vocab, cfg):
        super().__init__()
        d = cfg["d_mono"]
        self.enc = PrefixEncoder(vocab, d, cfg["enc_layers"], n_state=2)
        self.exp = Expert(d, cfg["exp_layers"], 2 * ARM_DIM)

    def _v(self, b):
        ctx, pad = self.enc(b["image"], b["instr"], [b["state_L"], b["state_R"]])
        return lambda x, t: self.exp(x, t, ctx, pad)

    def loss(self, b, gen=None):
        a = torch.cat([b["action_L"], b["action_R"]], -1)
        l = flow_loss(self._v(b), a, b["mask"], gen).mean()
        return {"loss": l, "bc": l}

    @torch.no_grad()
    def sample(self, b, gen=None):
        a = flow_sample(self._v(b), (b["image"].shape[0], H, 2 * ARM_DIM), b["image"].device, gen)
        return a[..., :ARM_DIM], a[..., ARM_DIM:]


class TwinVLA(nn.Module):
    """TwinVLA-like: two per-arm streams, both see the global instruction; joint attention across the
    streams in every encoder layer and always-on cross-arm attention in the experts (A-014)."""

    def __init__(self, vocab, cfg):
        super().__init__()
        d = cfg["d"]
        self.enc = nn.ModuleDict({k: PrefixEncoder(vocab, d, cfg["enc_layers"], 1) for k in "LR"})
        self.exp = nn.ModuleDict({k: Expert(d, cfg["exp_layers"], ARM_DIM, arm_xattn=True) for k in "LR"})

    def encode(self, b):
        xs, pads = {}, {}
        for k in "LR":
            xs[k], pads[k] = self.enc[k].embed(b["image"], b["instr"], [b[f"state_{k}"]])
        for bl, br in zip(self.enc["L"].blocks, self.enc["R"].blocks):
            new_l = bl.attend(xs["L"], xs["R"], torch.cat([pads["L"], pads["R"]], 1))
            new_r = br.attend(xs["R"], xs["L"], torch.cat([pads["R"], pads["L"]], 1))
            xs["L"], xs["R"] = new_l + bl.mlp(bl.n2(new_l)), new_r + br.mlp(br.n2(new_r))
        return self.enc["L"].out(xs["L"]), pads["L"], self.enc["R"].out(xs["R"]), pads["R"]

    def _v(self, b):
        cl, pl, cr, pr = self.encode(b)
        ones = torch.ones(b["image"].shape[0], device=b["image"].device)

        def v(x, t):
            vl, vr = paired_forward(self.exp["L"], self.exp["R"], x[..., :ARM_DIM], x[..., ARM_DIM:], t,
                                    cl, pl, cr, pr, ones)
            return torch.cat([vl, vr], -1)
        return v

    def loss(self, b, gen=None):
        a = torch.cat([b["action_L"], b["action_R"]], -1)
        l = flow_loss(self._v(b), a, b["mask"], gen).mean()
        return {"loss": l, "bc": l}

    @torch.no_grad()
    def sample(self, b, gen=None):
        a = flow_sample(self._v(b), (b["image"].shape[0], H, 2 * ARM_DIM), b["image"].device, gen)
        return a[..., :ARM_DIM], a[..., ARM_DIM:]


class CoopEstimator(nn.Module):
    """High-level encoder (image + global instruction; A-019) + transformer decoder with one learned query."""

    def __init__(self, vocab, d, enc_layers, dec_layers=2):
        super().__init__()
        self.high = PrefixEncoder(vocab, d, enc_layers, n_state=0)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.dec = nn.ModuleList(Block(d, cross=True) for _ in range(dec_layers))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, img, ids):
        z_h, pad = self.high(img, ids, [])
        q = self.query.expand(img.shape[0], -1, -1)
        for blk in self.dec:
            q = blk(q, ctx=z_h, ctx_pad=pad)
        return self.head(q)[:, 0, 0]  # logit


class SkillVLA(nn.Module):
    """Per-arm prompts (u_L, u_R) from the high-level planner -> per-arm prefix encoders (own state only,
    A-009) -> per-arm experts with cooperation-gated cross-arm attention (discrete gate, Sec. V-B)."""

    def __init__(self, vocab, cfg, gate_mode="learned"):
        super().__init__()
        d = cfg["d"]
        self.gate_mode = gate_mode
        self.enc = nn.ModuleDict({k: PrefixEncoder(vocab, d, cfg["enc_layers"], 1) for k in "LR"})
        self.exp = nn.ModuleDict({k: Expert(d, cfg["exp_layers"], ARM_DIM, arm_xattn=gate_mode != "off")
                                  for k in "LR"})
        if gate_mode == "learned":
            self.coop = CoopEstimator(vocab, d, cfg["high_layers"])
        self.lam = cfg["lambda"]

    def conditions(self, b):
        cl, pl = self.enc["L"](b["image"], b["u_L"], [b["state_L"]])
        cr, pr = self.enc["R"](b["image"], b["u_R"], [b["state_R"]])
        return cl, pl, cr, pr

    def gate_prob(self, img, ids):
        return torch.sigmoid(self.coop(img, ids).float())

    def _run(self, cond, x_l, x_r, tau, gate):
        return paired_forward(self.exp["L"], self.exp["R"], x_l, x_r, tau, *cond, gate)

    def loss(self, b, gen=None):
        B, dev = b["image"].shape[0], b["image"].device
        cond = self.conditions(b)
        tau = sample_tau(B, dev, gen)
        eps_l = torch.randn(b["action_L"].shape, device=dev, generator=gen)
        eps_r = torch.randn(b["action_R"].shape, device=dev, generator=gen)
        x_l, u_l = noisy(b["action_L"], eps_l, tau)
        x_r, u_r = noisy(b["action_R"], eps_r, tau)
        m = b["mask"]

        def per_sample(gate):
            vl, vr = self._run(cond, x_l, x_r, tau, gate)
            err = ((vl - u_l).pow(2).mean(-1) + (vr - u_r).pow(2).mean(-1))
            return (err * m).sum(-1) / m.sum(-1).clamp(min=1)

        zeros, ones = torch.zeros(B, device=dev), torch.ones(B, device=dev)
        if self.gate_mode == "off":
            l = per_sample(zeros).mean()
            return {"loss": l, "bc": l}
        if self.gate_mode == "on":
            l = per_sample(ones).mean()
            return {"loss": l, "bc": l}
        # learned discrete gate: same tau / noise / observation for both branches (A-005, A-023)
        l_on, l_off = per_sample(ones), per_sample(zeros)
        y_hat = self.gate_prob(b["image"], b["instr"])
        g = (y_hat.detach() >= 0.5).float()
        bc = (g * l_on + (1 - g) * l_off).mean()
        y = (l_on.detach() < l_off.detach()).float()
        with torch.no_grad():
            y_prev = self.gate_prob(b["prev"]["image"], b["prev"]["instr"])  # A-020: detached
        lam = self.lam
        with torch.autocast("cuda", enabled=False):
            yh = y_hat.float().clamp(1e-4, 1 - 1e-4)
            l_disc = F.binary_cross_entropy(yh, y)
            l_prior = F.binary_cross_entropy(yh, b["prior"].float())
            l_sticky = F.binary_cross_entropy(yh, y_prev.float())
            l_sup = yh.mean()
        gate_loss = lam["disc"] * l_disc + lam["prior"] * l_prior + lam["sticky"] * l_sticky + lam["sup"] * l_sup
        return {"loss": bc + gate_loss, "bc": bc, "l_on": l_on.mean(), "l_off": l_off.mean(),
                "y_frac": y.mean(), "gate_frac": g.mean(), "l_disc": l_disc, "l_prior": l_prior}

    @torch.no_grad()
    def sample(self, b, gen=None):
        B, dev = b["image"].shape[0], b["image"].device
        cond = self.conditions(b)
        if self.gate_mode == "learned":
            gate = (self.gate_prob(b["image"], b["instr"]) >= 0.5).float()
        else:
            gate = torch.full((B,), float(self.gate_mode == "on"), device=dev)

        def v(x, t):
            vl, vr = self._run(cond, x[..., :ARM_DIM], x[..., ARM_DIM:], t, gate)
            return torch.cat([vl, vr], -1)
        a = flow_sample(v, (B, H, 2 * ARM_DIM), dev, gen)
        return a[..., :ARM_DIM], a[..., ARM_DIM:], gate


DEFAULT_CFG = {"d": 128, "d_mono": 192, "enc_layers": 3, "exp_layers": 4, "high_layers": 2,
               "lambda": {"disc": 1.0, "prior": 1.0, "sticky": 0.1, "sup": 0.01}}  # A-008


def build(name, vocab, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    return {
        "mono": lambda: Mono(vocab, cfg),
        "twin": lambda: TwinVLA(vocab, cfg),
        "skillvla": lambda: SkillVLA(vocab, cfg, "learned"),
        "skillvla_noattn": lambda: SkillVLA(vocab, cfg, "off"),
    }[name]()
