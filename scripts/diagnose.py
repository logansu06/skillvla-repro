"""Train (or load) one policy and print per-episode failure details for one task."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.run import DEV, all_sentences  # noqa: E402
from skillvla import env as E  # noqa: E402
from skillvla.data import ChunkDataset, generate_demos  # noqa: E402
from skillvla.evaluate import EXEC_STEPS, to_batch  # noqa: E402
from skillvla.models import build  # noqa: E402
from skillvla.planner import coop_prior, rule_planner  # noqa: E402
from skillvla.text import Tokenizer  # noqa: E402
from skillvla.train import train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="skillvla")
    p.add_argument("--task", default="R_cake")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--train", default=",".join(E.SINGLE), help="comma-separated training tasks")
    a = p.parse_args()
    tok = Tokenizer(all_sentences())
    torch.manual_seed(0)
    pol = build(a.method, len(tok)).to(DEV)
    if os.path.exists(a.ckpt):
        pol.load_state_dict(torch.load(a.ckpt))
    else:
        eps = generate_demos([E.TASKS[n] for n in a.train.split(",")], 50, seed=0, coop_prior=coop_prior, planner=rule_planner)
        train(pol, ChunkDataset(eps, tok, DEV), a.steps, 256, 3e-4, 0, log_every=5000)
        os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
        torch.save(pol.state_dict(), a.ckpt)
    pol.eval()
    gen = torch.Generator(device=DEV).manual_seed(0)
    for i in range(a.n):
        env = E.BimanualEnv(E.TASKS[a.task], 10_000 * 1_000_003 + 3 + i)
        obs, done, events = env.reset(), False, []
        held_prev = {n: None for n in env.obj}
        while not done:
            b = to_batch([obs], tok, rule_planner, DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = pol.sample(b, gen)
            al, ar = (x.float().cpu().numpy()[0] for x in out[:2])
            for h in range(EXEC_STEPS):
                obs, done, info = env.step(al[h], ar[h])
                for n, o in env.obj.items():
                    if o["held"] != held_prev[n]:
                        events.append(f"t{env.t}:{n}{'+' if o['held'] else '-'}@{np.round(o['pos'][:2], 2)}")
                        held_prev[n] = o["held"]
                if done:
                    break
        recep = {k: np.round(v["pos"][:2], 2) for k, v in env.obj.items() if not v["grasp"]}
        if env.task.coop:
            recep = {k: v for k, v in env.mem["coop"].items() if k in ("target_y", "release_t", "grasped")}
            recep["final"] = {k: np.round(v["pos"], 2) for k, v in env.obj.items()}
        print(f"ep{i:02d} ok={info['success']} t={env.t} eeR={np.round(env.ee['R'], 2)} eeL={np.round(env.ee['L'], 2)} "
              f"recep={recep} events={events}")


if __name__ == "__main__":
    main()
