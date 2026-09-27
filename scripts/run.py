"""Entry point: generate demos -> train -> closed-loop eval -> write results.

  --smoke                 tiny end-to-end run, writes out/smoke.json only
  --exp recomposition     Table I / II analogue  -> results/recomposition.json
  --exp cooperative       Table III analogue     -> results/cooperative.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from skillvla import env as E  # noqa: E402
from skillvla.data import ChunkDataset, generate_demos  # noqa: E402
from skillvla.evaluate import evaluate  # noqa: E402
from skillvla.models import build  # noqa: E402
from skillvla.planner import coop_prior, rule_planner  # noqa: E402
from skillvla.text import Tokenizer  # noqa: E402
from skillvla.train import train  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = "cuda"


def all_sentences():
    out = []
    for t in E.TASKS.values():
        env = E.BimanualEnv(t, 0)
        env.reset()
        g = env.instruction()
        out += [g, *rule_planner(g)]
    return out


def smoke():
    tok = Tokenizer(all_sentences())
    train_tasks = [E.TASKS["L_cup"], E.TASKS["R_cake"]]
    eps = generate_demos(train_tasks, 4, seed=0, coop_prior=coop_prior, planner=rule_planner)
    ds = ChunkDataset(eps, tok, DEV)
    torch.manual_seed(0)
    pol = build("skillvla", len(tok)).to(DEV)
    train(pol, ds, steps=20, batch_size=32, lr=1e-3, seed=0, log_every=10)
    res = evaluate(pol, E.TASKS["cupxcake"], 2, seed=0, tok=tok, planner=rule_planner, device=DEV)
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    out = {"PLACEHOLDER_20step_skillvla_cupxcake": res, "n_train_frames": ds.n}
    with open(os.path.join(ROOT, "out", "smoke.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


EXPS = {
    # name: (list of (train_tasks, eval_cells) groups, default methods)
    "recomposition": ([(E.SINGLE, E.SINGLE + E.COMBOS)], ["mono", "twin", "skillvla"]),
    "cooperative": ([([c], [c]) for c in E.COOP], ["mono", "twin", "skillvla", "skillvla_noattn"]),
}


def experiment(name, methods, seeds, steps, eval_n, n_demos, out_path, eval_only=False):
    groups, default_methods = EXPS[name]
    methods = methods or default_methods
    tok = Tokenizer(all_sentences())
    runs, hist = {}, {}
    if eval_only and os.path.exists(out_path):
        with open(out_path) as f:
            hist = json.load(f).get("train_history", {})
    for train_names, eval_names in groups:
        for seed in seeds:
            if not eval_only:
                eps = generate_demos([E.TASKS[n] for n in train_names], n_demos, seed=seed,
                                     coop_prior=coop_prior, planner=rule_planner)
                ds = ChunkDataset(eps, tok, DEV)
            for m in methods:
                t0 = time.time()
                ck = os.path.join(ROOT, "out", "ckpt",
                                  f"{name}_{'+'.join(train_names) if len(groups) > 1 else 'all'}_{m}_s{seed}.pt")
                torch.manual_seed(seed)
                pol = build(m, len(tok)).to(DEV)
                if eval_only:
                    print(f"[{name}] method={m} seed={seed} train={train_names} frames=0 (eval-only: {os.path.basename(ck)})",
                          flush=True)
                    pol.load_state_dict(torch.load(ck))
                    pol.eval()
                else:
                    print(f"[{name}] method={m} seed={seed} train={train_names} frames={ds.n}", flush=True)
                    hist.setdefault(m, {})[f"{seed}:{'+'.join(train_names)}"] = train(
                        pol, ds, steps=steps, batch_size=256, lr=3e-4, seed=seed, log_every=max(steps // 10, 1))
                    os.makedirs(os.path.dirname(ck), exist_ok=True)
                    torch.save(pol.state_dict(), ck)
                for ci, cell in enumerate(eval_names):
                    # eval seeds are disjoint from demo seeds (demo seeds are < 100_000 * len(seeds))
                    res = evaluate(pol, E.TASKS[cell], eval_n, seed=10_000 + 97 * seed + ci, tok=tok,
                                   planner=rule_planner, device=DEV, record_gate=m == "skillvla")
                    runs.setdefault(m, {}).setdefault(str(seed), {})[cell] = res
                    print(f"   {cell:12s} {res['success']}/{res['n']}" +
                          (f" gate_on={res['gate_on_fraction']:.2f}" if res.get("gate_on_fraction") is not None else ""),
                          flush=True)
                print(f"   ({time.time() - t0:.0f}s)", flush=True)
                del pol
                torch.cuda.empty_cache()
    out = {"experiment": name, "methods": methods, "seeds": seeds, "steps": steps, "eval_n": eval_n,
           "n_demos_per_task": n_demos, "runs": runs, "train_history": hist,
           "evaluated_from_saved_checkpoints": eval_only}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--exp", choices=list(EXPS))
    p.add_argument("--methods", type=lambda s: s.split(","), default=None)
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])  # A-011
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--eval-n", type=int, default=50)  # A-010
    p.add_argument("--demos", type=int, default=50)   # A-010
    p.add_argument("--pilot", action="store_true", help="write to out/pilot_<exp>.json instead of results/")
    p.add_argument("--eval-only", action="store_true", help="re-evaluate saved checkpoints in out/ckpt/ (no training)")
    args = p.parse_args()
    if args.smoke:
        return smoke()
    if not args.exp:
        p.error("--smoke or --exp required")
    path = os.path.join(ROOT, "out" if args.pilot else "results",
                        f"{'pilot_' if args.pilot else ''}{args.exp}.json")
    experiment(args.exp, args.methods, args.seeds, args.steps, args.eval_n, args.demos, path, args.eval_only)


if __name__ == "__main__":
    main()
