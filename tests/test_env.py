import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skillvla.env import COMBOS, COOP, IMG, SINGLE, TASKS, BimanualEnv  # noqa: E402
from skillvla.planner import rule_planner  # noqa: E402


def run(task, seed, perturb=None):
    e = BimanualEnv(task, seed)
    e.reset()
    done, info = False, None
    while not done:
        a_l, a_r = e.expert()
        if perturb:
            a_l, a_r = perturb(e, a_l, a_r)
        _, done, info = e.step(a_l, a_r)
    return info["success"], e


@pytest.mark.parametrize("name", SINGLE + COMBOS + COOP)
def test_expert_solves_task(name):
    ok = sum(run(TASKS[name], s)[0] for s in range(40))
    assert ok >= 38, f"{name}: expert {ok}/40"


@pytest.mark.parametrize("name", SINGLE)
def test_single_arm_scene_has_no_objects_on_idle_side(name):  # A-004
    e = BimanualEnv(TASKS[name], 0)
    e.reset()
    idle_right = TASKS[name].right == "idle"
    for o in e.obj.values():
        assert (o["pos"][0] < 0.5) == idle_right


def test_idle_arm_must_stay_home():
    def drift(e, a_l, a_r):
        return a_l, np.array([0.0, 1.0, 0.0, -1.0])  # right arm drifts away
    assert not run(TASKS["L_cup"], 0, drift)[0]


@pytest.mark.parametrize("name", ["shake", "ball"])
def test_coop_requires_synchrony(name):
    def desync(e, a_l, a_r):
        if e.grip["L"] > 0 and e.grip["R"] > 0:
            a_r = a_r.copy()
            a_r[2] *= 0.5  # right arm half as fast in z
        return a_l, a_r
    assert sum(run(TASKS[name], s, desync)[0] for s in range(20)) == 0


def test_align_conflict_fails():
    def same_block(e, a_l, a_r):
        c = e.mem["coop"]
        if "assign" in c and c["assign"] and c.get("stage", 0) <= 1:
            c["assign"]["R"] = c["assign"]["L"]
        return a_l, a_r
    assert sum(run(TASKS["align"], s, same_block)[0] for s in range(20)) == 0


def test_obs_and_planner():
    seen = set()
    for name, t in TASKS.items():
        o = BimanualEnv(t, 0).reset()
        assert o["image"].shape == (IMG, IMG, 3) and o["image"].dtype == np.uint8
        assert o["state_L"].shape == (7,)
        u_l, u_r = rule_planner(o["instruction"])
        assert u_l and u_r
        seen.add(o["instruction"])
    assert len(seen) == len(TASKS)


def test_align_requires_shared_height():  # A-017 (revised): blocks must be collinear with each other
    def disagree(e, a_l, a_r):
        c = e.mem["coop"]
        if "goal" in c and not c.get("_shifted"):
            c["goal"]["R"] = c["goal"]["R"] + [0.0, 0.1]
            c["_shifted"] = True
        return a_l, a_r
    assert sum(run(TASKS["align"], s, disagree)[0] for s in range(20)) == 0


def test_stir_requires_a_real_circle():  # sweep round 1: back-and-forth must not count as stirring
    def wiggle(e, a_l, a_r):
        c = e.obj["bowl"]["pos"]
        m = e.mem["R"]
        if e.ee["R"][2] > 0.25:  # get above the bowl and descend
            d = np.r_[(c[:2] - e.ee["R"][:2]) / 0.04, -1.0]
            return a_l, np.r_[np.clip(d, -1, 1), -1.0]
        m["dir"] = -m.get("dir", 1) if abs(e.ee["R"][0] - c[0]) > 0.05 else m.get("dir", 1)
        return a_l, np.array([m["dir"] * 1.0, 0.0, 0.0, -1.0])  # sweep left-right across the bowl
    assert sum(run(TASKS["R_stir"], s, wiggle)[0] for s in range(10)) == 0


def _stir_on_path(offsets):
    """Feed end-effector positions (relative to the bowl centre, z=0.2) straight into the Stir check."""
    e = BimanualEnv(TASKS["R_stir"], 0)
    e.reset()
    c = e.obj["bowl"]["pos"]
    for dx, dy in offsets:
        e.ee["R"] = np.array([c[0] + dx, c[1] + dy, 0.2])
        if e._check_stir("R"):
            return True
    return False


def test_stir_metric_on_explicit_paths():  # sweep round 2: the exact diameter-crossing counterexample
    r = 0.04
    circle = [(r * np.cos(t), r * np.sin(t)) for t in np.linspace(0, 2.2 * np.pi, 40)]
    diameter_jumps = [(-r, 0.0), (r, 0.0)] * 20            # 180 deg per step, through the centre
    through_centre = [(x, 0.0) for x in np.linspace(-r, r, 9)] * 10
    offset_wiggle = [(x, 0.02) for x in list(np.linspace(-r, r, 9)) + list(np.linspace(r, -r, 9))] * 10
    assert _stir_on_path(circle)
    assert not _stir_on_path(diameter_jumps)
    assert not _stir_on_path(through_centre)
    assert not _stir_on_path(offset_wiggle)
