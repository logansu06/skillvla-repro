"""Demo generation and chunked dataset held on GPU."""
import numpy as np
import torch

from .env import BimanualEnv
from .text import Tokenizer

H = 8  # action chunk length


def run_expert_episode(task, seed):
    env = BimanualEnv(task, seed)
    obs = env.reset()
    frames = []
    done, info = False, {"success": False}
    while not done:
        a_l, a_r = env.expert()
        frames.append((obs, a_l, a_r))
        obs, done, info = env.step(a_l, a_r)
    return frames, info["success"]


def generate_demos(tasks, n_per_task, seed, coop_prior=None, planner=None):
    """Return a list of episodes; only successful expert episodes are kept."""
    episodes, s = [], seed * 100_000
    for task in tasks:
        kept = 0
        while kept < n_per_task:
            frames, ok = run_expert_episode(task, s)
            s += 1
            if not ok:
                continue
            kept += 1
            u_l, u_r = planner(frames[0][0]["instruction"]) if planner else ("", "")
            episodes.append({"task": task, "frames": frames, "u_L": u_l, "u_R": u_r,
                             "prior": float(coop_prior(task)) if coop_prior else 0.0})
    return episodes


class ChunkDataset:
    """Every frame of every episode becomes one sample with an H-step action chunk."""

    def __init__(self, episodes, tok: Tokenizer, device):
        imgs, sL, sR, aL, aR, mask, instr, uL, uR, prior, prev = ([] for _ in range(11))
        idx = 0
        for ep in episodes:
            T = len(ep["frames"])
            acts_l = np.stack([f[1] for f in ep["frames"]])
            acts_r = np.stack([f[2] for f in ep["frames"]])
            ids_g = tok.encode(ep["frames"][0][0]["instruction"])
            ids_l, ids_r = tok.encode(ep["u_L"]), tok.encode(ep["u_R"])
            for t, (obs, _, _) in enumerate(ep["frames"]):
                imgs.append(obs["image"])
                sL.append(obs["state_L"]); sR.append(obs["state_R"])
                sl = slice(t, min(t + H, T))
                pad = H - (sl.stop - sl.start)
                aL.append(np.concatenate([acts_l[sl], np.repeat(acts_l[-1:], pad, 0)]))
                aR.append(np.concatenate([acts_r[sl], np.repeat(acts_r[-1:], pad, 0)]))
                mask.append(np.r_[np.ones(H - pad), np.zeros(pad)])
                instr.append(ids_g); uL.append(ids_l); uR.append(ids_r)
                prior.append(ep["prior"])
                prev.append(idx + max(t - 1, 0))
            idx += T
        f = lambda x, dt=torch.float32: torch.as_tensor(np.stack(x), dtype=dt, device=device)
        self.t = {
            "image": f(imgs, torch.uint8), "state_L": f(sL), "state_R": f(sR),
            "action_L": f(aL), "action_R": f(aR), "mask": f(mask),
            "instr": f(instr, torch.long), "u_L": f(uL, torch.long), "u_R": f(uR, torch.long),
            "prior": f(prior), "prev": f(prev, torch.long),
        }
        self.n = len(imgs)

    def sample(self, batch_size, gen):
        i = torch.randint(0, self.n, (batch_size,), device=self.t["image"].device, generator=gen)
        b = {k: v[i] for k, v in self.t.items() if k != "prev"}
        j = self.t["prev"][i]
        b["prev"] = {k: self.t[k][j] for k in ("image", "instr", "state_L", "state_R")}
        return b
