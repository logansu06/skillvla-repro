"""Rule-based high-level planner (A-003) and rule-based cooperation prior (A-006)."""
import re

from .env import COOP_TEXT

COOP_PROMPTS = {g: (ul, ur) for g, ul, ur in COOP_TEXT.values()}  # global instruction -> (u_L, u_R)


def rule_planner(instruction):
    """Decompose a global instruction into per-arm skill prompts (u_L, u_R)."""
    if instruction in COOP_PROMPTS:
        return COOP_PROMPTS[instruction]
    m = re.fullmatch(r"left arm : (.+?) \. right arm : (.+?)", instruction)
    if m is None:
        raise ValueError(f"planner cannot parse: {instruction!r}")
    return m.group(1), m.group(2)


def coop_prior(task):
    return 1.0 if task.coop else 0.0
