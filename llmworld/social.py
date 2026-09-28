"""Emotions, relationships, and turning them into an inner voice.

Numbers live in the sim; the prompt only ever sees prose written as the agent's
own thoughts ("Tom insulted you. You want Tom to pay for it."), never labels.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .entities import EMOTIONS, Agent

if TYPE_CHECKING:
    from .sim import Simulation

CAUSE_MEMORY = 400  # ticks a cause stays attached to a feeling

# How a listener can feel about what someone said to them. The LLM picks one;
# the effects are fixed so the sim stays predictable.
FELT = {
    "insulted": {"emo": {"anger": 0.2, "joy": -0.05}, "rel": {"affinity": -0.15, "respect": -0.03}, "reason": "insulted you"},
    "threatened": {"emo": {"fear": 0.2, "anger": 0.1}, "rel": {"fear": 0.2, "affinity": -0.1, "trust": -0.1}, "reason": "threatened you"},
    "respected": {"emo": {"joy": 0.1}, "rel": {"affinity": 0.08, "respect": 0.04}, "reason": "treated you with respect"},
    "flattered": {"emo": {"joy": 0.15}, "rel": {"affinity": 0.12}, "reason": "flattered you"},
    "persuaded": {"emo": {"joy": 0.05}, "rel": {"respect": 0.1, "trust": 0.06}, "reason": "made a lot of sense"},
    "helped": {"emo": {"joy": 0.15}, "rel": {"affinity": 0.15, "trust": 0.12}, "reason": "helped you"},
    "betrayed": {"emo": {"anger": 0.3, "sadness": 0.1}, "rel": {"trust": -0.4, "affinity": -0.2}, "reason": "betrayed you"},
    "comforted": {"emo": {"sadness": -0.2, "joy": 0.1}, "rel": {"affinity": 0.12, "trust": 0.05}, "reason": "comforted you"},
    "neutral": {"emo": {}, "rel": {"affinity": 0.01}, "reason": ""},
}

REL_BOUNDS = {"affinity": (-1.0, 1.0), "trust": (-1.0, 1.0), "fear": (0.0, 1.0), "respect": (0.0, 1.0)}


def _mods(sim: Simulation, a: Agent) -> dict:
    return sim.trait_mods(a)


def feel(sim: Simulation, a: Agent, emotion: str, delta: float, source: str | None = None, reason: str | None = None) -> None:
    if emotion not in a.emotions or not delta:
        return
    delta *= _mods(sim, a)["sensitivity"].get(emotion, 1.0)
    a.emotions[emotion] = max(0.0, min(1.0, a.emotions[emotion] + delta))
    if delta > 0 and reason:
        a.causes.append({"emotion": emotion, "source": source, "reason": reason, "tick": sim.tick, "strength": delta})
        del a.causes[:-24]


def relate(sim: Simulation, a: Agent, other_id: str, field: str, delta: float) -> None:
    if other_id == a.id or other_id not in sim.agents:
        return
    mods = _mods(sim, a)
    if field == "trust":
        delta *= mods["trust_gain"] if delta > 0 else mods["trust_loss"]
    r = a.rel(other_id)
    lo, hi = REL_BOUNDS[field]
    setattr(r, field, max(lo, min(hi, getattr(r, field) + delta)))
    r.met = True


def react(sim: Simulation, a: Agent, speaker_id: str, felt: str) -> None:
    fx = FELT.get(felt)
    if not fx:
        return
    speaker = sim.agents[speaker_id]
    for emo, d in fx["emo"].items():
        feel(sim, a, emo, d, speaker.name, fx["reason"])
    for field, d in fx["rel"].items():
        relate(sim, a, speaker_id, field, d)
    if felt != "neutral":
        a.log(sim.tick, "feel", f"felt {felt} by {speaker.name}")


def decay(sim: Simulation, a: Agent) -> None:
    base = _mods(sim, a)["baseline"]
    for e in EMOTIONS:
        a.emotions[e] += (base.get(e, 0.0) - a.emotions[e]) * 0.012


def on_event(sim: Simulation, a: Agent, ev, role: str) -> None:
    """Emotional and relational fallout of something `a` experienced. role: target | witness."""
    k = ev.kind
    actor = sim.agents.get(ev.actor) if ev.actor else None
    target = sim.agents.get(ev.target) if ev.target else None
    if k == "attack" and actor:
        if role == "target":
            feel(sim, a, "anger", 0.3, actor.name, "attacked you")
            feel(sim, a, "fear", 0.25, actor.name, "attacked you")
            relate(sim, a, actor.id, "affinity", -0.3)
            relate(sim, a, actor.id, "trust", -0.3)
            relate(sim, a, actor.id, "fear", 0.25)
        elif target:
            relate(sim, a, actor.id, "fear", 0.08)
            if a.rel(target.id).affinity > 0.2:
                relate(sim, a, actor.id, "affinity", -0.15)
                feel(sim, a, "anger", 0.12, actor.name, f"attacked {target.name}")
            else:
                relate(sim, a, actor.id, "respect", 0.03)
    elif k == "gift" and actor:
        if role == "target":
            feel(sim, a, "joy", 0.2, actor.name, f"gave you {ev.data['qty']} {ev.data['item']}")
            relate(sim, a, actor.id, "affinity", 0.2)
            relate(sim, a, actor.id, "trust", 0.1)
        else:
            relate(sim, a, actor.id, "affinity", 0.03)
            relate(sim, a, actor.id, "respect", 0.03)
    elif k == "theft" and actor:
        if role == "target":
            feel(sim, a, "anger", 0.3, actor.name, "stole from you")
            relate(sim, a, actor.id, "trust", -0.4)
            relate(sim, a, actor.id, "affinity", -0.25)
        else:
            relate(sim, a, actor.id, "trust", -0.2)
    elif k == "seize" and actor and role == "target":
        feel(sim, a, "anger", 0.35, actor.name, f"took your {ev.data['kind']}")
        relate(sim, a, actor.id, "affinity", -0.3)
        relate(sim, a, actor.id, "trust", -0.3)
    elif k == "death" and target:
        grief = max(0.0, a.rel(target.id).affinity) * 0.7 + 0.1
        feel(sim, a, "sadness", grief, target.name, "is dead")
        feel(sim, a, "fear", 0.15, target.name, "is dead")
        if actor:
            relate(sim, a, actor.id, "fear", 0.3)
            relate(sim, a, actor.id, "affinity", -0.3 if a.rel(target.id).affinity > -0.2 else 0.05)
            feel(sim, a, "fear", 0.2, actor.name, f"killed {target.name}")
    elif k == "monster_attack":
        if role == "target":
            feel(sim, a, "fear", 0.35, "A monster", "attacked you")
        else:
            feel(sim, a, "fear", 0.12, "A monster", f"attacked {target.name if target else 'someone'}")
    elif k == "farm_raid":
        if role == "target":
            feel(sim, a, "anger", 0.15, "A monster", "is destroying your crops")
            feel(sim, a, "fear", 0.1, "A monster", "is destroying your crops")
        else:
            feel(sim, a, "fear", 0.08, "A monster", "is loose near the farms")
    elif k == "monster_killed" and actor:
        if role == "witness":
            relate(sim, a, actor.id, "respect", 0.15)
            feel(sim, a, "joy", 0.1, actor.name, "killed a monster")
    elif k == "built" and actor and role == "witness":
        relate(sim, a, actor.id, "respect", 0.04)
    elif k == "helped_build" and actor and role == "target":
        feel(sim, a, "joy", 0.12, actor.name, "helped you build")
        relate(sim, a, actor.id, "affinity", 0.12)
        relate(sim, a, actor.id, "trust", 0.08)
    elif k == "request_answer" and actor and role == "target":
        if ev.data.get("accept"):
            feel(sim, a, "joy", 0.06, actor.name, "agreed to do what you asked")
        else:
            feel(sim, a, "anger", 0.12, actor.name, "refused your request")
            relate(sim, a, actor.id, "affinity", -0.05)


# --- prose -----------------------------------------------------------------


def _level(value: float, thresholds: list[float], descending: bool = False) -> int:
    lvl = -1
    for i, t in enumerate(thresholds):
        if (value <= t) if descending else (value >= t):
            lvl = i
    return lvl


def _cause(sim: Simulation, a: Agent, emotion: str) -> dict | None:
    best, score = None, 0.0
    for c in a.causes:
        if c["emotion"] != emotion or sim.tick - c["tick"] > CAUSE_MEMORY:
            continue
        s = c["strength"] * (1.0 - (sim.tick - c["tick"]) / CAUSE_MEMORY)
        if s > score:
            best, score = c, s
    return best


def feelings(sim: Simulation, a: Agent, rng: random.Random | None = None) -> list[str]:
    """The agent's strongest emotions, in their own inner voice."""
    ph = sim.data["phrases"]["emotions"]
    out = []
    ranked = sorted(EMOTIONS, key=lambda e: a.emotions[e], reverse=True)
    for e in ranked[:2]:
        spec = ph[e]
        lvl = _level(a.emotions[e], spec["levels"])
        if lvl < 0:
            continue
        c = _cause(sim, a, e)
        if c and c["source"]:
            out.append(spec["with_source"][lvl].format(source=c["source"], reason=c["reason"]))
        else:
            out.append(spec["generic"][lvl])
    return out


def body(sim: Simulation, a: Agent, water_hint: str, food_hint: str) -> list[str]:
    """Needs and health, as sensations."""
    ph = sim.data["phrases"]["needs"]
    out = []
    for need, value, hint in (("thirst", a.thirst, water_hint), ("hunger", a.hunger, food_hint), ("fatigue", a.fatigue, "")):
        lvl = _level(value, ph[need]["levels"])
        if lvl >= 0:
            out.append(ph[need]["text"][lvl].format(hint=hint).strip())
    lvl = _level(a.hp, ph["health"]["levels"], descending=True)
    if lvl >= 0:
        out.append(ph["health"]["text"][lvl])
    return out


def relation_hint(sim: Simulation, a: Agent, other: Agent) -> str:
    ph = sim.data["phrases"]["relation"]
    r = a.relations.get(other.id)
    if not r or not r.met:
        return ph["stranger"].format(name=other.name)
    bits = []
    if r.fear > 0.4:
        bits.append(ph["fear_high"])
    if r.affinity > 0.45:
        bits.append(ph["affinity_high"])
    elif r.affinity < -0.35:
        bits.append(ph["affinity_low"])
    if r.trust < -0.35:
        bits.append(ph["trust_low"])
    elif r.trust > 0.5:
        bits.append(ph["trust_high"])
    if r.respect > 0.5:
        bits.append(ph["respect_high"])
    return "; ".join(b.format(name=other.name) for b in bits[:2])
