"""Closed-loop evaluation with batched policy inference over parallel envs."""
import numpy as np
import torch

from .env import BimanualEnv

EXEC_STEPS = 4  # receding horizon: execute the first 4 of H=8 actions (A-013)


def to_batch(obs_list, tok, planner, device):
    f = lambda k, dt=torch.float32: torch.as_tensor(np.stack([o[k] for o in obs_list]), dtype=dt, device=device)
    instr = [o["instruction"] for o in obs_list]
    u = [planner(s) for s in instr]
    ids = lambda xs: torch.tensor([tok.encode(s) for s in xs], dtype=torch.long, device=device)
    return {"image": f("image", torch.uint8), "state_L": f("state_L"), "state_R": f("state_R"),
            "instr": ids(instr), "u_L": ids([x[0] for x in u]), "u_R": ids([x[1] for x in u])}


@torch.no_grad()
def evaluate(policy, task, n, seed, tok, planner, device, record_gate=False):
    envs = [BimanualEnv(task, seed * 1_000_003 + i) for i in range(n)]
    obs = [e.reset() for e in envs]
    done = [False] * n
    info = [{"success": False, "success_L": False, "success_R": False}] * n
    gates = [[] for _ in range(n)]
    gen = torch.Generator(device=device).manual_seed(seed)
    while not all(done):
        live = [i for i in range(n) if not done[i]]
        b = to_batch([obs[i] for i in live], tok, planner, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = policy.sample(b, gen)
        a_l, a_r = (x.float().cpu().numpy() for x in out[:2])
        if record_gate and len(out) > 2:
            for j, i in enumerate(live):
                gates[i].append(float(out[2][j]))
        for j, i in enumerate(live):
            for h in range(EXEC_STEPS):
                obs[i], done[i], info[i] = envs[i].step(a_l[j, h], a_r[j, h])
                if done[i]:
                    break
    res = {"success": int(sum(x["success"] for x in info)), "n": n,
           "success_L": int(sum(x["success_L"] for x in info)),
           "success_R": int(sum(x["success_R"] for x in info))}
    if record_gate:
        res["gate_on_fraction"] = float(np.mean([np.mean(g) for g in gates if g])) if any(gates) else None
    return res
