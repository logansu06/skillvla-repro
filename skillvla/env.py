"""Planar (top-down, with height) bimanual tabletop simulator.

Workspace is [0,1]^3 (x right, y away from robot, z height). The left arm owns the
left half of the table, the right arm the right half. Each arm is an end effector
with a binary gripper. Actions per arm: (dx, dy, dz, grip) in [-1, 1], scaled by
MAX_DXY / MAX_DZ; grip > 0 closes.

Single-arm skills (names follow the paper; geometry is our analogue, A-002):
  left : cup (pick cup -> place on plate), box (push box over goal line), mug (grasp + lift + hold)
  right: cake (pick cake -> place in container), stir (circle inside bowl), smash (strike nut from high)
Cooperative tasks (A-017): shake, ball, align.
"""
from dataclasses import dataclass

import numpy as np

IMG = 64
MAX_DXY, MAX_DZ = 0.04, 0.10
HOME = {"L": np.array([0.2, 0.15, 0.6]), "R": np.array([0.8, 0.15, 0.6])}
SIDE_X = {"L": (0.10, 0.40), "R": (0.60, 0.90)}
GRASP_R, GRASP_Z = 0.045, 0.12
IDLE_TOL = 0.10

PHRASE = {
    "cup": "place the cup on the plate", "box": "push the box to the line", "mug": "lift the mug",
    "cake": "put the cake in the box", "stir": "stir the bowl", "smash": "smash the nut",
    "idle": "stay",
}
LEFT_SKILLS, RIGHT_SKILLS = ("cup", "box", "mug"), ("cake", "stir", "smash")
COOP_TEXT = {
    "shake": ("shake the capped cup with both arms", "shake the cup", "hold the cap"),
    "ball": ("lift the ball with both arms", "lift the ball from the left", "lift the ball from the right"),
    "align": ("align the two blocks with both arms", "grasp a block and align it", "grasp a block and align it"),
}

COLORS = {
    "cup": (230, 40, 40), "plate": (245, 245, 245), "box": (140, 90, 40), "line": (240, 220, 0),
    "mug": (255, 140, 0), "cake": (255, 255, 255), "container": (40, 80, 255), "bowl": (40, 200, 60),
    "nut": (255, 200, 0), "cupcap": (200, 30, 30), "cap": (250, 250, 120), "ball": (255, 255, 255),
    "blockA": (220, 40, 40), "blockB": (160, 160, 0),
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    left: str        # skill id for the left arm, "idle", or the coop task name
    right: str
    coop: bool = False


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


class BimanualEnv:
    def __init__(self, task: TaskSpec, seed: int, horizon: int | None = None):
        self.task = task
        self.horizon = horizon or (120 if task.coop else 100)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ setup
    def reset(self):
        r = self.rng
        self.t = 0
        self.ee = {k: HOME[k] + np.r_[r.uniform(-0.02, 0.02, 2), 0.0] for k in "LR"}
        self.prev = {k: np.zeros(3) for k in "LR"}
        self.grip = {k: -1.0 for k in "LR"}
        self.obj = {}            # name -> {"pos": xyz, "grasp": bool, "held": None|"L"|"R", "part_of": ...}
        self.flag = {"L": False, "R": False}
        self.failed = False
        self.mem = {"L": {}, "R": {}, "coop": {}}
        self.speed = {k: r.uniform(0.75, 1.0) for k in "LR"}  # expert speed factor per episode
        if self.task.coop:
            getattr(self, f"_setup_{self.task.left}")()
        else:
            for k, s in (("L", self.task.left), ("R", self.task.right)):
                if s == "idle":
                    self.flag[k] = True  # an idle arm has nothing to achieve; idleness checked in success()
                else:
                    getattr(self, f"_setup_{s}")(k)
        return self.obs()

    def _free_xy(self, k, n, min_sep=0.16):
        lo, hi = SIDE_X[k]
        while True:
            pts = [np.array([self.rng.uniform(lo, hi), self.rng.uniform(0.32, 0.78)]) for _ in range(n)]
            if all(_dist(p, q) >= min_sep for i, p in enumerate(pts) for q in pts[i + 1:]):
                return pts

    def _add(self, name, xy, grasp=False, z=0.0):
        self.obj[name] = {"pos": np.array([xy[0], xy[1], z]), "grasp": grasp, "held": None}

    def _setup_cup(self, k):
        a, b = self._free_xy(k, 2)
        self._add("cup", a, grasp=True); self._add("plate", b)

    def _setup_cake(self, k):
        a, b = self._free_xy(k, 2)
        self._add("cake", a, grasp=True); self._add("container", b)

    def _setup_mug(self, k):
        (a,) = self._free_xy(k, 1)
        self._add("mug", a, grasp=True)

    def _setup_box(self, k):
        lo, hi = SIDE_X[k]
        self._add("box", [self.rng.uniform(lo + 0.03, hi - 0.03), self.rng.uniform(0.35, 0.6)])

    def _setup_stir(self, k):
        (a,) = self._free_xy(k, 1)
        self._add("bowl", a)
        self.mem[k]["angle"], self.mem[k]["last_ang"] = 0.0, None

    def _setup_smash(self, k):
        (a,) = self._free_xy(k, 1)
        self._add("nut", a)
        self.mem[k]["z_hist"] = []

    def _setup_two_part(self, a_name, b_name, half_gap):
        c = np.array([0.5, self.rng.uniform(0.4, 0.7)])
        self._add(a_name, c - [half_gap, 0], grasp=True)
        self._add(b_name, c + [half_gap, 0], grasp=True)
        self.obj[a_name]["part_of"] = self.obj[b_name]["part_of"] = "pair"
        self.mem["coop"]["gap"] = 2 * half_gap

    def _setup_shake(self):
        self._setup_two_part("cupcap", "cap", 0.05)
        self.mem["coop"].update(amp=self.rng.uniform(0.08, 0.16), period=int(self.rng.integers(10, 17)),
                                crossings=0, side=0, center=0.4)

    def _setup_ball(self):
        self._setup_two_part("ball", "ballR", 0.06)
        self.mem["coop"]["v"] = self.rng.uniform(0.02, 0.07)

    def _setup_align(self):
        while True:
            pa = np.array([self.rng.uniform(0.42, 0.58), self.rng.uniform(0.3, 0.75)])
            pb = np.array([self.rng.uniform(0.42, 0.58), self.rng.uniform(0.3, 0.75)])
            if _dist(pa, pb) > 0.12 and abs(pa[0] - pb[0]) >= 0.05:
                break
        self._add("blockA", pa, grasp=True); self._add("blockB", pb, grasp=True)
        # block positions are randomized, so the arm-object assignment varies across trials (A-017, rev. 2)
        self.mem["coop"].update(place_y=(pa[1] + pb[1]) / 2, release_t={}, assign=None)

    # ------------------------------------------------------------- observation
    def instruction(self):
        if self.task.coop:
            return COOP_TEXT[self.task.left][0]
        return f"left arm : {PHRASE[self.task.left]} . right arm : {PHRASE[self.task.right]}"

    def state(self, k):
        return np.concatenate([self.ee[k], [self.grip[k]], self.prev[k]]).astype(np.float32)

    def render(self):
        img = np.full((IMG, IMG, 3), 80, np.uint8)
        yy, xx = np.mgrid[0:IMG, 0:IMG]

        def disk(xy, rad, col, ring=False):
            cx, cy = xy[0] * (IMG - 1), xy[1] * (IMG - 1)
            d = np.hypot(xx - cx, yy - cy)
            m = (d <= rad) & (d >= rad - 1.2) if ring else d <= rad
            img[m] = col

        if "box" in self.obj:
            img[int(0.85 * (IMG - 1)), int(0.08 * IMG):int(0.44 * IMG)] = COLORS["line"]
        order = sorted(self.obj, key=lambda n: self.obj[n]["grasp"])  # receptacles first
        for n in order:
            o = self.obj[n]
            if n in ("plate", "bowl", "container"):
                disk(o["pos"], 4.5, COLORS[n], ring=True)
            else:
                disk(o["pos"], 2.0 + 3.0 * o["pos"][2], COLORS.get(n, (255, 255, 255)))
        for k, col in (("L", (0, 255, 255)), ("R", (255, 0, 255))):
            rad = 1.0 + 3.0 * self.ee[k][2]
            disk(self.ee[k], rad, col, ring=True)
            if self.grip[k] > 0:
                disk(self.ee[k], 0.8, (0, 0, 0))
        return img

    def obs(self):
        return {"image": self.render(), "state_L": self.state("L"), "state_R": self.state("R"),
                "instruction": self.instruction()}

    # ------------------------------------------------------------------ physics
    def step(self, a_l, a_r):
        for k, a in (("L", a_l), ("R", a_r)):
            a = np.clip(np.asarray(a, np.float64), -1, 1)
            d = a[:3] * np.array([MAX_DXY, MAX_DXY, MAX_DZ])
            old = self.ee[k].copy()
            self.ee[k] = np.clip(self.ee[k] + d, 0, 1)
            self._push_box(old, self.ee[k])
            self.prev[k] = a[:3]
            new_grip = 1.0 if a[3] > 0 else -1.0
            if new_grip > 0 and self.grip[k] < 0:
                self._try_grasp(k)
            elif new_grip < 0 and self.grip[k] > 0:
                self._release(k)
            self.grip[k] = new_grip
        for o in self.obj.values():
            if o["held"] is not None:
                o["pos"] = self.ee[o["held"]].copy()
        self.t += 1
        self._update_flags()
        success = self.success()
        done = success or self.failed or self.t >= self.horizon
        info = {"success": success, "success_L": self.flag["L"], "success_R": self.flag["R"]}
        return self.obs(), done, info

    def _try_grasp(self, k):
        if any(o["held"] == k for o in self.obj.values()):
            return
        cands = [(n, _dist(o["pos"], self.ee[k])) for n, o in self.obj.items()
                 if o["grasp"] and o["held"] is None]
        cands = [c for c in cands if c[1] < GRASP_R and self.ee[k][2] < GRASP_Z]
        if cands:
            n = min(cands, key=lambda c: c[1])[0]
            self.obj[n]["held"] = k
            if self.task.coop and self.task.left == "align":
                self.mem["coop"].setdefault("grasped", {})[k] = n

    def _release(self, k):
        for n, o in self.obj.items():
            if o["held"] == k:
                o["held"] = None
                o["pos"][2] = 0.0
                if self.task.coop and self.task.left == "align":
                    self.mem["coop"]["release_t"][k] = (self.t, n)

    def _push_box(self, old, new):
        """Non-prehensile push: a low end effector moving into the box drags it along its motion."""
        if "box" not in self.obj or new[2] >= GRASP_Z:
            return
        box = self.obj["box"]["pos"]
        delta = new[:2] - old[:2]
        step = np.linalg.norm(delta)
        if step < 1e-9:
            return
        rel = box[:2] - old[:2]
        along = rel @ delta / step
        lateral = abs(rel[0] * delta[1] - rel[1] * delta[0]) / step
        if 0 < along < 0.045 + step and lateral < 0.03:
            box[:2] = np.clip(box[:2] + delta * (along + step - 0.045).clip(0, step) / step, 0, 1)

    # ---------------------------------------------------------------- success
    def _idle_ok(self, k):
        return _dist(self.ee[k], HOME[k]) < IDLE_TOL and abs(self.ee[k][2] - HOME[k][2]) < 0.2

    def _update_flags(self):
        if self.task.coop:
            return getattr(self, f"_check_{self.task.left}")()
        for k, s in (("L", self.task.left), ("R", self.task.right)):
            if s != "idle" and not self.flag[k]:
                self.flag[k] = getattr(self, f"_check_{s}")(k)

    def _placed(self, item, recep, tol):
        o = self.obj[item]
        return o["held"] is None and o["pos"][2] == 0.0 and _dist(o["pos"], self.obj[recep]["pos"]) < tol

    def _check_cup(self, k):
        return self._placed("cup", "plate", 0.05)

    def _check_cake(self, k):
        return self._placed("cake", "container", 0.05)

    def _check_box(self, k):
        return self.obj["box"]["pos"][1] >= 0.85

    def _check_mug(self, k):
        m = self.mem[k]
        up = self.obj["mug"]["held"] == k and self.ee[k][2] >= 0.6
        m["hold"] = m.get("hold", 0) + 1 if up else 0
        return m["hold"] >= 5

    def _check_stir(self, k):
        m, c, e = self.mem[k], self.obj["bowl"]["pos"], self.ee[k]
        # count angle only on a ring around the bowl centre; crossing the centre breaks the track
        if e[2] < 0.3 and 0.015 < _dist(e, c) < 0.08:
            ang = np.arctan2(e[1] - c[1], e[0] - c[0])
            if m["last_ang"] is not None:
                dth = (ang - m["last_ang"] + np.pi) % (2 * np.pi) - np.pi  # signed: reversals cancel
                if abs(dth) < np.pi / 2:  # a >=90 deg jump in one step is not continuous circling
                    m["angle"] += dth
            m["last_ang"] = ang
        else:
            m["last_ang"] = None
        return abs(m["angle"]) >= 2 * np.pi  # one full circuit in either direction

    def _check_smash(self, k):
        m, e = self.mem[k], self.ee[k]
        m["z_hist"] = (m["z_hist"] + [e[2]])[-10:]
        return e[2] <= 0.1 and _dist(e, self.obj["nut"]["pos"]) < 0.04 and max(m["z_hist"]) >= 0.6

    def _pair_ok(self, a, b, ztol, xytol):
        pa, pb = self.obj[a]["pos"], self.obj[b]["pos"]
        dz = abs(pa[2] - pb[2])
        dxy = abs((pb[0] - pa[0]) - self.mem["coop"]["gap"]) + abs(pb[1] - pa[1])
        return dz < ztol and dxy < xytol

    def _both_hold(self, a, b):
        return self.obj[a]["held"] == "L" and self.obj[b]["held"] == "R"

    def _check_shake(self):
        c = self.mem["coop"]
        if max(self.obj["cupcap"]["pos"][2], self.obj["cap"]["pos"][2]) > 0.02 and \
                not self._pair_ok("cupcap", "cap", 0.05, 0.05):
            self.failed = True  # cap separated from cup
            return
        if self._both_hold("cupcap", "cap"):
            dz = self.ee["L"][2] - c["center"]
            side = 1 if dz > 0.05 else (-1 if dz < -0.05 else 0)
            if side and side != c["side"]:
                if c["side"]:
                    c["crossings"] += 1
                c["side"] = side
        self.flag["L"] = self.flag["R"] = c["crossings"] >= 6

    def _check_ball(self):
        if max(self.obj["ball"]["pos"][2], self.obj["ballR"]["pos"][2]) > 0.02 and \
                not self._pair_ok("ball", "ballR", 0.04, 0.05):
            self.failed = True  # ball tilted / dropped
            return
        ok = self._both_hold("ball", "ballR") and min(self.ee["L"][2], self.ee["R"][2]) >= 0.75
        self.flag["L"] = self.flag["R"] = ok

    def _check_align(self):
        c = self.mem["coop"]
        rel = c["release_t"]
        if len(rel) < 2:
            return
        (tl, nl), (tr, nr) = rel["L"], rel["R"]
        pl, pr = self.obj[nl]["pos"], self.obj[nr]["pos"]
        ok = (nl != nr and abs(tl - tr) <= 3 and abs(pl[1] - pr[1]) < 0.05
              and pl[0] < 0.45 and pr[0] > 0.55)
        self.flag["L"] = self.flag["R"] = ok
        if not ok:
            self.failed = True

    def success(self):
        if self.task.coop:
            return self.flag["L"] and self.flag["R"] and not self.failed
        ok = True
        for k, s in (("L", self.task.left), ("R", self.task.right)):
            ok &= self._idle_ok(k) if s == "idle" else self.flag[k]
        return ok

    # --------------------------------------------------------- scripted experts
    def _move(self, k, target, grip, speed=None, zspeed=None):
        """Action toward target; returns (action, reached)."""
        d = target - self.ee[k]
        sp = self.speed[k] if speed is None else speed
        zsp = sp if zspeed is None else zspeed
        a = np.r_[np.clip(d[:2] / MAX_DXY, -sp, sp), np.clip(d[2] / MAX_DZ, -zsp, zsp)]
        a[:2] += self.rng.normal(0, 0.03, 2)
        reached = np.abs(d[:2]).max() < 0.008 and abs(d[2]) < 0.008
        return np.r_[a, grip], reached

    def _waypoints(self, k, wps):
        """Follow a list of (target_xyz, grip, zspeed) waypoints; phase kept in mem."""
        m = self.mem[k]
        i = m.get("wp", 0)
        if i >= len(wps):
            tgt, g, zs = wps[-1]
            return self._move(k, tgt, g, zspeed=zs)[0]
        tgt, g, zs = wps[i]
        a, reached = self._move(k, tgt, g, zspeed=zs)
        if reached:
            m["wp"] = i + 1
        return a

    def _pick_place_wps(self, item, recep):
        o, r = self.obj[item]["pos"], self.obj[recep]["pos"]
        return [((o[0], o[1], 0.4), -1, None), ((o[0], o[1], 0.05), -1, 0.5), ((o[0], o[1], 0.05), 1, None),
                ((o[0], o[1], 0.4), 1, None), ((r[0], r[1], 0.4), 1, None), ((r[0], r[1], 0.1), 1, 0.5),
                ((r[0], r[1], 0.1), -1, None), ((r[0], r[1], 0.4), -1, None)]

    def _expert_arm(self, k, skill):
        m = self.mem[k]
        if skill == "idle":
            a, _ = self._move(k, HOME[k], -1)
            a[:2] = np.clip(a[:2], -0.3, 0.3)
            return a
        if "wps" not in m:  # plan once at the first call, from the initial object layout
            if skill in ("cup", "cake"):
                m["wps"] = self._pick_place_wps(skill, "plate" if skill == "cup" else "container")
            elif skill == "mug":
                o = self.obj["mug"]["pos"]
                m["wps"] = [((o[0], o[1], 0.4), -1, None), ((o[0], o[1], 0.05), -1, 0.5),
                            ((o[0], o[1], 0.05), 1, None), ((o[0], o[1], 0.7), 1, None)]
            elif skill == "box":
                b = self.obj["box"]["pos"]
                m["wps"] = [((b[0], b[1] - 0.07, 0.3), -1, None), ((b[0], b[1] - 0.07, 0.05), -1, 0.5),
                            ((b[0], 0.84, 0.05), -1, None), ((b[0], 0.84, 0.4), -1, None)]
            elif skill == "stir":
                c, rad = self.obj["bowl"]["pos"], 0.04
                circle = [((c[0] + rad * np.cos(th), c[1] + rad * np.sin(th), 0.2), -1, None)
                          for th in np.linspace(0, 2.5 * np.pi, 21)]
                m["wps"] = [((c[0] + rad, c[1], 0.5), -1, None)] + circle + [((c[0], c[1], 0.5), -1, None)]
            elif skill == "smash":
                n = self.obj["nut"]["pos"]
                m["wps"] = [((n[0], n[1], 0.8), -1, None), ((n[0], n[1], 0.02), 1, 1.0),
                            ((n[0], n[1], 0.4), 1, None)]
        return self._waypoints(k, m["wps"])

    def _expert_coop(self):
        c, name = self.mem["coop"], self.task.left
        if "stage" not in c:
            c["stage"], c["k"] = 0, 0
            if name == "align":  # allocation: the left arm takes the block further left
                a, b = sorted(("blockA", "blockB"), key=lambda n: self.obj[n]["pos"][0])
                c["assign"] = {"L": a, "R": b}
                ty = c["place_y"]
                c["goal"] = {"L": np.array([0.3, ty]), "R": np.array([0.7, ty])}
            else:
                a, b = ("cupcap", "cap") if name == "shake" else ("ball", "ballR")
                c["assign"] = {"L": a, "R": b}
        pos = {k: self.obj[c["assign"][k]]["pos"] for k in "LR"}
        st, sync = c["stage"], 1.0  # both arms share one speed in coop demos

        def both(targets, grip, zspeed=None):
            acts, reached = {}, True
            for k in "LR":
                a, r = self._move(k, np.asarray(targets[k], float), grip, speed=sync, zspeed=zspeed)
                acts[k], reached = a, reached and r
            return acts, reached

        # stage 0: above grasp points; 1: descend; 2: close; 3+: task-specific
        if st == 0:
            acts, r = both({k: (*pos[k][:2], 0.3) for k in "LR"}, -1)
        elif st == 1:
            acts, r = both({k: (*pos[k][:2], 0.05) for k in "LR"}, -1, zspeed=0.5)
        elif st == 2:
            acts, r = both({k: self.ee[k] for k in "LR"}, 1)
            r = True
        elif name == "shake":
            if st == 3:
                acts, r = both({k: (*self.ee[k][:2], c["center"]) for k in "LR"}, 1)
            else:
                c["k"] += 1
                z = c["center"] + c["amp"] * np.sin(2 * np.pi * c["k"] / c["period"])
                acts, _ = both({k: (*self.ee[k][:2], z) for k in "LR"}, 1)
                r = False
        elif name == "ball":
            v = c["v"] / MAX_DZ
            acts = {k: np.array([0.0, 0.0, v if self.ee[k][2] < 0.8 else 0.0, 1.0]) for k in "LR"}
            r = False
        else:  # align: lift, carry to target line, lower, release together
            g = c["goal"]
            plan = {3: ({k: (*self.ee[k][:2], 0.3) for k in "LR"}, 1, None),
                    4: ({k: (*g[k], 0.3) for k in "LR"}, 1, None),
                    5: ({k: (*g[k], 0.05) for k in "LR"}, 1, 0.5),
                    6: ({k: self.ee[k] for k in "LR"}, -1, None),
                    7: ({k: (*g[k], 0.3) for k in "LR"}, -1, None)}
            tg, gr, zs = plan[min(st, 7)]
            acts, r = both(tg, gr, zs)
            if st == 6:
                r = True
        if r:
            c["stage"] = st + 1
        return acts["L"], acts["R"]

    def expert(self):
        if self.task.coop:
            return self._expert_coop()
        return self._expert_arm("L", self.task.left), self._expert_arm("R", self.task.right)


def _build_tasks():
    t = {}
    for s in LEFT_SKILLS:
        t[f"L_{s}"] = TaskSpec(f"L_{s}", s, "idle")
    for s in RIGHT_SKILLS:
        t[f"R_{s}"] = TaskSpec(f"R_{s}", "idle", s)
    for a in LEFT_SKILLS:
        for b in RIGHT_SKILLS:
            t[f"{a}x{b}"] = TaskSpec(f"{a}x{b}", a, b)
    for c in COOP_TEXT:
        t[c] = TaskSpec(c, c, c, coop=True)
    return t


TASKS = _build_tasks()
SINGLE = [f"L_{s}" for s in LEFT_SKILLS] + [f"R_{s}" for s in RIGHT_SKILLS]
COMBOS = [f"{a}x{b}" for a in LEFT_SKILLS for b in RIGHT_SKILLS]
COOP = list(COOP_TEXT)
