#!/usr/bin/env python3
"""Live terminal dashboard for scripts/run.py experiments.

  .venv/bin/python scripts/monitor.py                  # watch every running run.py, refresh every 5 s
  .venv/bin/python scripts/monitor.py -n 2             # refresh every 2 s
  .venv/bin/python scripts/monitor.py --detail         # also list every finished model
  .venv/bin/python scripts/monitor.py --once           # print one snapshot and exit
  .venv/bin/python scripts/monitor.py --log out/x.log  # also show a (finished) log

Running jobs are found via /proc: the log is whatever the process's stdout points to.
"""
import argparse
import glob
import math
import os
import re
import subprocess
import time
from collections import OrderedDict

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# mirrors EXPS / argparse defaults in scripts/run.py
DEFAULT_METHODS = {"recomposition": ["mono", "twin", "skillvla"],
                   "cooperative": ["mono", "twin", "skillvla", "skillvla_noattn"]}
GROUPS = {"recomposition": ["all"], "cooperative": ["shake", "ball", "align"]}
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_STEPS = 20_000

EXP_TITLE = {"recomposition": "组合实验 · Table I / II", "cooperative": "协作实验 · Table III"}
METHOD = {"mono": ("Mono (π0.5-like)", "bright_blue"), "twin": ("TwinVLA-like", "magenta"),
          "skillvla": ("SkillVLA", "bold green"), "skillvla_noattn": ("SkillVLA w/o Attn", "yellow")}
PAPER = {  # paper's real-robot numbers, for side-by-side reading only
    "recomposition": {"mono": (0.77, 0.00), "twin": (0.67, 0.04), "skillvla": (0.78, 0.51)},
    "cooperative": {"mono": (0.30, 0.45, 0.65), "twin": (0.15, 0.55, 0.55), "skillvla": (0.25, 0.50, 0.70),
                    "skillvla_noattn": (0.00, 0.10, 0.40)},
}

RE_START = re.compile(r"^\[(\w+)\] method=(\S+) seed=(\d+) train=\[(.*)\] frames=(\d+)")
RE_STEP = re.compile(r"^\s+step (\d+)/(\d+) (.*) \((\d+)s\)\s*$")
RE_EVAL = re.compile(r"^\s{3}(\S+)\s+(\d+)/(\d+)(?: gate_on=([\d.]+))?\s*$")
RE_DONE = re.compile(r"^\s{3}\((\d+)s\)\s*$")


# ------------------------------------------------------------------ helpers
def fmt_t(s):
    if s is None:
        return "?"
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def method_text(m):
    name, style = METHOD.get(m, (m, "white"))
    return Text(name, style=style)


def rate_text(v, suffix=""):
    if v is None:
        return Text("—", style="bright_black")
    style = "bold green" if v >= 0.7 else "green" if v >= 0.5 else "yellow" if v >= 0.3 else \
        "dark_orange" if v >= 0.1 else "bold red"
    return Text.assemble((f"{v:.2f}", style), (suffix, "bright_black"))


def spark(vals, width=36):
    vals = [v for v in vals if v is not None and v > 0][-width:]
    if len(vals) < 2:
        return Text("")
    lv = [math.log10(v) for v in vals]
    lo, hi = min(lv), max(lv)
    ch = "▁▂▃▄▅▆▇█"
    return Text("".join(ch[int((x - lo) / (hi - lo + 1e-12) * (len(ch) - 1))] for x in lv), style="cyan")


def bar(done, total, width, style):
    return ProgressBar(total=max(total, 1e-9), completed=done, width=width, complete_style=style,
                       finished_style=style, style="grey23")


# ------------------------------------------------------------------ discovery & parsing
def running_jobs():
    jobs = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if not any(a.endswith("scripts/run.py") for a in argv) or "--smoke" in argv:
            continue
        try:
            log = os.readlink(f"/proc/{pid}/fd/1")
            start = os.stat(f"/proc/{pid}").st_mtime
        except OSError:
            continue
        jobs.append({"pid": int(pid), "argv": argv, "log": log, "started": start})
    return sorted(jobs, key=lambda j: j["started"])


def plan_from_argv(argv):
    def opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default
    exp = opt("--exp", None)
    methods = opt("--methods", None)
    methods = methods.split(",") if methods else DEFAULT_METHODS.get(exp, [])
    seeds = opt("--seeds", None)
    seeds = [int(s) for s in seeds.split(",")] if seeds else DEFAULT_SEEDS
    return {"exp": exp, "methods": methods, "seeds": seeds, "steps": int(opt("--steps", DEFAULT_STEPS)),
            "pilot": "--pilot" in argv, "groups": GROUPS.get(exp, ["all"])}


def parse_log(path):
    runs, cur, error, wrote = [], None, [], None
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return {"runs": [], "error": [f"cannot read {path}"], "wrote": None}
    in_tb = False
    for ln in lines:
        if ln.startswith("Traceback"):
            in_tb = True
        if in_tb:
            error.append(ln)
            continue
        m = RE_START.match(ln)
        if m:
            cur = {"exp": m[1], "method": m[2], "seed": int(m[3]),
                   "train": [t.strip().strip("'") for t in m[4].split(",")], "frames": int(m[5]),
                   "steps": [], "evals": OrderedDict(), "done_s": None}
            runs.append(cur)
            continue
        if cur is None:
            continue
        m = RE_STEP.match(ln)
        if m:
            kv = dict(re.findall(r"(\w+)=([-\d.eE+]+)", m[3]))
            cur["steps"].append({"step": int(m[1]), "total": int(m[2]), "t": int(m[4]),
                                 **{k: float(v) for k, v in kv.items()}})
            continue
        m = RE_EVAL.match(ln)
        if m:
            cur["evals"][m[1]] = (int(m[2]), int(m[3]), float(m[4]) if m[4] else None)
            continue
        m = RE_DONE.match(ln)
        if m:
            cur["done_s"] = int(m[1])
            continue
        if ln.startswith("wrote "):
            wrote = ln[6:]
    return {"runs": runs, "error": error, "wrote": wrote}


def group_of(run):
    return "all" if len(run["train"]) > 1 else run["train"][0]


def split_rates(evals):
    """Recomposition: (seen-skill rate, unseen-combo rate) pooled over cells."""
    def pooled(cells):
        n = sum(evals[c][1] for c in cells)
        return sum(evals[c][0] for c in cells) / n if n else None
    seen = [c for c in evals if c[:2] in ("L_", "R_")]
    combo = [c for c in evals if c not in seen]
    return pooled(seen), pooled(combo), len(seen), len(combo)


# ------------------------------------------------------------------ widgets
def gpu_panel():
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_column(justify="right")
    grid.add_column()
    grid.add_column(justify="right")
    grid.add_column(justify="right", style="bright_black")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,"
                            "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        q = []
    for ln in q:
        name, util, used, tot, temp, pw = [x.strip() for x in ln.split(",")]
        util, used, tot = int(util), int(used) / 1024, int(tot) / 1024
        ustyle = "green" if util >= 70 else "yellow" if util >= 30 else "red"
        grid.add_row(name.replace("NVIDIA GeForce ", ""), bar(util, 100, 18, ustyle), f"{util}%",
                     bar(used, tot, 18, "cyan"), f"{used:.1f}/{tot:.0f} GB", f"{temp}°C  {float(pw):.0f} W")
    if not q:
        grid.add_row("GPU", Text("nvidia-smi 不可用", style="bright_black"), "", "", "", "")
    title = Text.assemble(("SkillVLA 复现 · 训练监控", "bold white"), "   ",
                          (time.strftime("%Y-%m-%d %H:%M:%S"), "bright_black"))
    return Panel(grid, title=title, title_align="left", box=box.ROUNDED, border_style="bright_black",
                 subtitle=Text("GPU 利用率 · 显存", style="bright_black"), subtitle_align="right")


def current_panel(cur, exp):
    g = Table.grid(padding=(0, 1))
    g.add_column(style="bright_black", no_wrap=True)
    g.add_column()
    g.add_row("方法", Text.assemble(method_text(cur["method"]), (f"   seed {cur['seed']}", "white")))
    g.add_row("训练集", Text(f"{group_of(cur)}  ·  {cur['frames']} 帧"))
    if not cur["steps"]:
        g.add_row("状态", Text("准备数据 / 等待首条日志 …", style="bright_black"))
        return Panel(g, title="当前模型", title_align="left", border_style="cyan", box=box.ROUNDED)
    st = cur["steps"][-1]
    if len(cur["steps"]) >= 2:
        a, b = cur["steps"][-2], cur["steps"][-1]
        sp = (b["step"] - a["step"]) / max(b["t"] - a["t"], 1e-9)
    else:
        sp = st["step"] / max(st["t"], 1e-9)
    training = st["step"] < st["total"]
    prog = Table.grid(padding=(0, 1))
    prog.add_row(bar(st["step"], st["total"], 26, "cyan" if training else "green"),
                 Text(f"{st['step'] / st['total']:.0%}", style="bold"))
    g.add_row("训练" if training else "评测中", prog)
    g.add_row("", Text(f"{st['step']:,}/{st['total']:,} 步 · {sp:.1f} it/s · "
                       + (f"训练剩余 {fmt_t((st['total'] - st['step']) / sp)}" if training and sp else "训练完成"),
                       style="bright_black"))
    g.add_row("bc 走势", Text.assemble(spark([s.get("bc") for s in cur["steps"]]),
                                      (f"  {cur['steps'][0]['bc']:.3g} → {st['bc']:.3g}", "bright_black")))
    mt = Text()
    for k, lab in (("bc", "BC"), ("l_on", "L_on"), ("l_off", "L_off"), ("y_frac", "y=1")):
        if k in st:
            mt.append(f"{lab} ", style="bright_black")
            mt.append(f"{st[k]:.4f}  ", style="bold white" if k == "bc" else "white")
    g.add_row("指标", mt)
    if "gate_frac" in st:
        on = st["gate_frac"]
        g.add_row("协作门", Text(f"● 开 {on:.0%}", style="bold green") if on >= 0.5 else
                  Text(f"○ 关（开 {on:.0%}）", style="bright_black"))
    if cur["evals"]:
        if exp == "recomposition":
            s, c, ns, nc = split_rates(cur["evals"])
            g.add_row("已评测", Text.assemble("已学 ", rate_text(s, f" ({ns}/6)"), "   组合 ", rate_text(c, f" ({nc}/9)")))
        else:
            t = Text()
            for cell, (s, n, gate) in cur["evals"].items():
                t.append(f"{cell} ")
                t.append(rate_text(s / n, f" ({s}/{n})  "))
            g.add_row("已评测", t)
    return Panel(g, title="当前模型", title_align="left", border_style="cyan", box=box.ROUNDED)


def results_table(exp, done, detail):
    agg = OrderedDict()
    for r in done:
        a = agg.setdefault(r["method"], {"seeds": set(), "cells": {}, "gate": {}})
        a["seeds"].add(r["seed"])
        for c, (s, n, gate) in r["evals"].items():
            v = a["cells"].setdefault(c, [0, 0])
            v[0] += s
            v[1] += n
            if gate is not None:
                a["gate"].setdefault(c, []).append(gate)
    tb = Table(box=box.SIMPLE_HEAD, header_style="bold", show_edge=False, pad_edge=False,
               title="结果（已完成种子合并）", title_style="bold", title_justify="left")
    tb.add_column("方法", no_wrap=True)
    tb.add_column("种子", justify="center")
    paper = PAPER.get(exp, {})
    if exp == "recomposition":
        tb.add_column("已学技能", justify="right")
        tb.add_column("未见组合", justify="right")
        tb.add_column("论文", justify="right", style="bright_black", no_wrap=True)
        for m, a in agg.items():
            ev = {c: (v[0], v[1], None) for c, v in a["cells"].items()}
            s, c, _, _ = split_rates(ev)
            p = paper.get(m)
            tb.add_row(method_text(m), str(len(a["seeds"])), rate_text(s), rate_text(c),
                       f"{p[0]:.2f}/{p[1]:.2f}" if p else "")
        tb.caption = "论文列：已学 / 组合（真机）"
    else:
        tasks = GROUPS["cooperative"]
        for t in tasks:
            tb.add_column(t.capitalize(), justify="right")
        tb.add_column("平均", justify="right")
        tb.add_column("论文", justify="right", style="bright_black", no_wrap=True)
        for m, a in agg.items():
            vals = [a["cells"][t][0] / a["cells"][t][1] if t in a["cells"] else None for t in tasks]
            cells = []
            for t, v in zip(tasks, vals):
                gate = a["gate"].get(t)
                cells.append(rate_text(v, f" 门{sum(gate) / len(gate):.1f}" if gate else ""))
            got = [v for v in vals if v is not None]
            p = paper.get(m)
            tb.add_row(method_text(m), str(len(a["seeds"])), *cells,
                       rate_text(sum(got) / len(got)) if len(got) == len(tasks) else Text("…", style="bright_black"),
                       "/".join(f"{x:.2f}" for x in p) if p else "")
        tb.caption = "论文列：Shake / Ball / Align（真机）；门 = 回合内协作门开启比例"
    tb.caption_style = "bright_black"
    tb.caption_justify = "left"
    if not detail:
        return tb
    dt = Table(box=box.SIMPLE_HEAD, header_style="bold", show_edge=False, pad_edge=False,
               title="已完成模型", title_style="bold", title_justify="left")
    for col, j in (("方法", "left"), ("seed", "center"), ("训练集", "left"), ("用时", "right")):
        dt.add_column(col, justify=j, no_wrap=True)
    dt.add_column("结果")
    for r in done:
        if exp == "recomposition":
            s, c, _, _ = split_rates(r["evals"])
            res = Text.assemble("已学 ", rate_text(s), "  组合 ", rate_text(c))
        else:
            res = Text()
            for cell, (s, n, gate) in r["evals"].items():
                res.append(rate_text(s / n, f" ({s}/{n})"))
        dt.add_row(method_text(r["method"]), str(r["seed"]), group_of(r), fmt_t(r["done_s"]), res)
    return Group(tb, Text(""), dt)


def job_panel(job, detail):
    plan = plan_from_argv(job["argv"]) if job.get("argv") else None
    info = parse_log(job["log"])
    runs = info["runs"]
    exp = (plan or {}).get("exp") or (runs[0]["exp"] if runs else "?")
    alive = job.get("pid") is not None
    done = [r for r in runs if r["done_s"] is not None]
    if info["error"]:
        status, border = Text("✖ 出错", style="bold red"), "red"
    elif info["wrote"]:
        status, border = Text("✔ 完成", style="bold green"), "green"
    elif alive:
        status, border = Text("● 运行中", style="bold green"), "green"
    else:
        status, border = Text("○ 未运行", style="bright_black"), "bright_black"

    head = Table.grid(padding=(0, 2))
    head.add_row(status, Text(f"pid {job['pid']}" if alive else "", style="bright_black"),
                 Text(f"已运行 {fmt_t(time.time() - job['started'])}" if alive else "", style="bright_black"))
    parts = [head]

    if plan and plan["methods"]:
        order = [(g, s, m) for g in plan["groups"] for s in plan["seeds"] for m in plan["methods"]]
        dur = {}
        for r in done:
            dur.setdefault(r["method"], []).append(r["done_s"])
        mean_all = (sum(map(sum, dur.values())) / sum(map(len, dur.values()))) if dur else None
        remaining, eta, known = order[len(done):], 0.0, True
        for i, (_, _, m) in enumerate(remaining):
            e = sum(dur[m]) / len(dur[m]) if m in dur else mean_all
            if i == 0 and runs and runs[-1]["done_s"] is None and runs[-1]["steps"]:
                st = runs[-1]["steps"][-1]
                if e is None:
                    e = st["t"] * st["total"] / max(st["step"], 1)
                e = max(e - st["t"], 0)
            if e is None:
                known = False
                break
            eta += e
        ov = Table.grid(padding=(0, 1))
        ov.add_row(Text("总进度", style="bold"), bar(len(done), len(order), 40, "green"),
                   Text(f"{len(done)}/{len(order)} 个模型", style="bold"),
                   Text(f"· 预计剩余 {fmt_t(eta) if known else '?'}" if remaining else "· 全部完成",
                        style="yellow"))
        parts.append(ov)

    cur = runs[-1] if runs and runs[-1]["done_s"] is None else None
    body = Table.grid(padding=(0, 3), expand=False)
    left = current_panel(cur, exp) if cur and alive else None
    right = results_table(exp, done, detail) if done else Text("还没有完成的模型", style="bright_black")
    if left is not None:
        body.add_row(left, right)
    else:
        body.add_row(right)
    parts += [Text(""), body]

    if info["error"]:
        parts.append(Panel(Text("\n".join(info["error"][-8:]), style="red"), title="错误", border_style="red"))
    if info["wrote"]:
        wrote = os.path.relpath(info["wrote"], ROOT) if info["wrote"].startswith("/") else info["wrote"]
        parts.append(Text(f"✔ 结果已写入 {os.path.normpath(wrote)}", style="green"))

    tag = "pilot" if plan and plan["pilot"] else ("正式" if plan else "")
    title = Text.assemble(("▶ ", "bold"), (EXP_TITLE.get(exp, exp), "bold white"),
                          (f"  [{tag}]" if tag else "", "yellow" if tag == "pilot" else "cyan"))
    log = os.path.relpath(job["log"], ROOT) if job["log"].startswith(ROOT) else job["log"]
    return Panel(Group(*parts), title=title, title_align="left", subtitle=Text(log, style="bright_black"),
                 subtitle_align="right", border_style=border, box=box.ROUNDED, padding=(0, 1))


def dashboard(extra_logs, detail):
    jobs = running_jobs()
    seen = {os.path.realpath(j["log"]) for j in jobs}
    for p in extra_logs:
        if os.path.realpath(p) not in seen:
            jobs.append({"pid": None, "argv": None, "log": os.path.abspath(p), "started": None})
    parts = [gpu_panel()]
    if not jobs:
        logs = sorted(glob.glob(os.path.join(ROOT, "out", "*.log")), key=os.path.getmtime)
        hint = f"最近的日志：{os.path.relpath(logs[-1], ROOT)}（用 --log 查看）" if logs else ""
        parts.append(Panel(Text("没有正在运行的 run.py。" + hint, style="bright_black"), box=box.ROUNDED))
    parts += [job_panel(j, detail) for j in jobs]
    return Group(*parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--interval", type=float, default=5.0, help="刷新间隔（秒）")
    p.add_argument("--once", action="store_true", help="只输出一次")
    p.add_argument("--detail", action="store_true", help="列出每个已完成模型")
    p.add_argument("--log", action="append", default=[], help="额外显示的日志文件（可多次）")
    a = p.parse_args()
    console = Console()
    if a.once:
        console.print(dashboard(a.log, a.detail))
        return
    footer = lambda: Text(f" 每 {a.interval:g}s 刷新 · Ctrl-C 退出 · 窗口太矮时内容会被截断，可加 --once 或放大终端",
                          style="bright_black")
    try:
        with Live(Group(dashboard(a.log, a.detail), footer()), console=console, screen=True,
                  auto_refresh=False) as live:
            while True:
                time.sleep(a.interval)
                live.update(Group(dashboard(a.log, a.detail), footer()), refresh=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
