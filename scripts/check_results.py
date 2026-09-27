"""Validate a results JSON (schema + completeness) and write the markdown table next to it.

Exit code 0 only if every (method, seed, cell) is present with n == eval_n and no PLACEHOLDER key exists.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skillvla import env as E  # noqa: E402

CELLS = {"recomposition": E.SINGLE + E.COMBOS, "cooperative": E.COOP}
NAMES = {"mono": "Mono (π0.5-like)", "twin": "TwinVLA-like", "skillvla": "SkillVLA",
         "skillvla_noattn": "SkillVLA w/o Attn."}


PROTOCOL = {  # declared in implement-stage/ASSUMPTIONS.md (A-010, A-011) and scripts/run.py EXPS
    "recomposition": {"methods": ["mono", "twin", "skillvla"]},
    "cooperative": {"methods": ["mono", "twin", "skillvla", "skillvla_noattn"]},
}


def check(r):
    assert "PLACEHOLDER" not in json.dumps(r), "placeholder value in a results file"
    want = PROTOCOL[r["experiment"]]
    assert sorted(r["methods"]) == sorted(want["methods"]), ("methods", r["methods"])
    assert r["seeds"] == [0, 1, 2], ("seeds", r["seeds"])
    assert r["eval_n"] == 50 and r["n_demos_per_task"] == 50 and r["steps"] == 20_000, "protocol mismatch"
    cells = CELLS[r["experiment"]]
    for m in r["methods"]:
        for s in r["seeds"]:
            got = r["runs"][m][str(s)]
            missing = [c for c in cells if c not in got]
            assert not missing, (m, s, missing)
            for c in cells:
                assert got[c]["n"] == r["eval_n"] and 0 <= got[c]["success"] <= got[c]["n"], (m, s, c)


def rate(r, m, c):
    per_seed = [r["runs"][m][str(s)][c]["success"] / r["runs"][m][str(s)][c]["n"] for s in r["seeds"]]
    return float(np.mean(per_seed)), float(np.std(per_seed)), per_seed


def label(cell):
    if cell in E.COMBOS:
        a = next(x for x in E.LEFT_SKILLS if cell.startswith(x + "x"))
        return f"{a}×{cell[len(a) + 1:]}"
    return cell.replace("L_", "").replace("R_", "")


def table(r, cells, title):
    head = "| Method | " + " | ".join(label(c) for c in cells) + " | Avg. |"
    lines = [f"### {title}", "", head, "|" + "---|" * (len(cells) + 2)]
    for m in r["methods"]:
        vals = [rate(r, m, c) for c in cells]
        seed_avgs = np.mean([v[2] for v in vals], axis=0)
        row = [f"{v[0]:.2f}" for v in vals] + [f"**{seed_avgs.mean():.2f}** ± {seed_avgs.std():.2f}"]
        lines.append(f"| {NAMES.get(m, m)} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def markdown(r):
    hdr = (f"Experiment `{r['experiment']}` — {len(r['seeds'])} training seeds × {r['eval_n']} episodes per cell; "
           f"{r['n_demos_per_task']} demos per task; {r['steps']} training steps. Cell = mean success rate over "
           f"seeds; Avg = mean ± std over seeds of the per-seed average.\n")
    if r["experiment"] == "recomposition":
        body = [table(r, E.COMBOS, "Table I analogue — unseen skill recompositions (zero-shot)"),
                table(r, E.SINGLE, "Table II analogue — trained single-arm skills")]
    else:
        body = [table(r, E.COOP, "Table III analogue — cooperative tasks")]
    gate = []
    for m in r["methods"]:
        if m != "skillvla":
            continue
        for c in CELLS[r["experiment"]]:
            g = [r["runs"][m][str(s)][c].get("gate_on_fraction") for s in r["seeds"]]
            g = [x for x in g if x is not None]
            if g:
                gate.append(f"{c}: {np.mean(g):.2f}")
    if gate:
        body.append("SkillVLA mean gate-on fraction during rollouts — " + ", ".join(gate))
    return hdr + "\n" + "\n\n".join(body) + "\n"


def main(path):
    with open(path) as f:
        r = json.load(f)
    check(r)
    md = markdown(r)
    with open(os.path.splitext(path)[0] + ".md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1])
