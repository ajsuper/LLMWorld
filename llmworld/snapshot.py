"""JSON views of the simulation for the web UI (and anything else that wants to render it)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import prompts, social
from .entities import EMOTIONS

if TYPE_CHECKING:
    from .brain import Brain
    from .entities import Agent
    from .sim import Simulation


def init_payload(sim: Simulation) -> dict:
    w = sim.world
    return {
        "type": "init",
        "size": w.size,
        "tiles": ["".join(row) for row in w.grid],
        "landmarks": [{"name": lm.name, "x": lm.x, "y": lm.y, "kind": lm.kind, "hidden": lm.hidden} for lm in w.landmarks.values()],
        "bushes": [[b.id, b.x, b.y, b.berries] for b in w.bushes.values()],
        "agents": [
            {"id": a.id, "name": a.name, "color": a.color, "traits": a.traits,
             "ambition": sim.data["ambitions"][a.ambition].get("label", a.ambition) if a.ambition else None}
            for a in sim.agents.values()
        ],
        "structures": structures(sim),
        "feed": list(sim.feed)[-120:],
        "tick_seconds": sim.cfg["world"]["tick_seconds"],
        "day_length": sim.day_length,
        "night_fraction": sim.cfg["world"]["night_fraction"],
    }


def structures(sim: Simulation) -> list[dict]:
    out = []
    for s in sim.world.structures.values():
        out.append({"id": s.id, "kind": s.kind, "x": s.x, "y": s.y, "owner": s.owner, "complete": s.complete,
                    "progress": round(s.work_done / s.work_needed, 2) if s.work_needed else 1, "burning": s.burning,
                    "stock": s.stock, "items": sum(s.items.values()), "of": s.of,
                    "needs": s.needs_text(), "water": round(s.water, 2), "growth": round(s.growth, 2)})
    return out


def tick_payload(sim: Simulation, brain: Brain, paused: bool) -> dict:
    w = sim.world
    msg = {
        "type": "tick",
        "tick": sim.tick,
        "day": sim.day(),
        "tod": round(sim.time_of_day(), 4),
        "night": sim.is_night(),
        "phase": prompts.time_phrase(sim),
        "paused": paused,
        "agents": [
            {"id": a.id, "x": a.x, "y": a.y, "hp": round(a.hp, 2), "alive": a.alive, "sleeping": a.sleeping,
             "thinking": a.thinking, "queued": a.queued,
             "act": ((a.task.sub.label if a.task.sub else a.task.label) if a.task and a.task.status == "active" else "idle") if a.alive else f"dead ({a.death_cause})",
             "bubble": {"text": a.bubble[0], "volume": a.bubble[1]} if a.bubble else None}
            for a in sim.agents.values()
        ],
        "monsters": [[m.id, m.x, m.y] for m in sim.monsters.values()],
        "tiles": [[x, y, ch] for x, y, ch in w.tile_changes],
        "bushes": [[bid, w.bushes[bid].berries] for bid in w.bush_changes],
        "feed": sim.tick_feed,
        "brain": brain.stats(),
    }
    if w.structures_dirty:
        msg["structures"] = structures(sim)
    return msg


def agent_detail(sim: Simulation, a: Agent) -> dict:
    wh, fh = prompts._hints(sim, a)
    rels = []
    for oid, r in a.relations.items():
        if not r.met:
            continue
        o = sim.agents[oid]
        rels.append({"id": oid, "name": o.name, "color": o.color, "alive": o.alive,
                     "affinity": round(r.affinity, 2), "trust": round(r.trust, 2), "fear": round(r.fear, 2),
                     "respect": round(r.respect, 2), "obeyed": r.obeyed, "refused": r.refused,
                     "hint": social.relation_hint(sim, a, o)})
    rels.sort(key=lambda r: -(abs(r["affinity"]) + r["fear"] + r["respect"] + abs(r["trust"])))
    amb = sim.data["ambitions"].get(a.ambition) if a.ambition else None
    return {
        "type": "detail",
        "id": a.id,
        "name": a.name,
        "color": a.color,
        "alive": a.alive,
        "death_cause": a.death_cause,
        "traits": a.traits,
        "ambition": amb.get("label", a.ambition) if amb else None,
        "changes": [{"t": c["t"], "now": c["now"], "why": c["why"]} for c in a.past_purposes],
        "tier": a.tier,
        "pos": [a.x, a.y],
        "hp": round(a.hp, 3),
        "needs": {"thirst": round(a.thirst, 3), "hunger": round(a.hunger, 3), "fatigue": round(a.fatigue, 3)},
        "emotions": {e: round(a.emotions[e], 3) for e in EMOTIONS},
        "inner": social.feelings(sim, a) + social.body(sim, a, wh, fh) if a.alive else [],
        "purpose": prompts.purpose_reminder(sim, a) if a.alive else "",
        "group": a.group,
        "leads": bool(a.group and sim.groups[a.group].leader == a.id),
        "job": {"for": sim.agents[a.duty["for"]].name, "task": a.duty["task"]} if a.duty else None,
        "crew": [o.name for o in sim.alive() if o.duty and o.duty["for"] == a.id],
        "task": (a.task.label + (f", {a.task.sub.label}" if a.task.sub else "")) if a.task and a.task.status == "active" else "idle",
        "thinking": a.thinking,
        "queued": a.queued,
        "think_reason": a.think_reason,
        "think_count": a.think_count,
        "inventory": {k: v for k, v in a.inventory.items() if v},
        "notepad": list(a.notepad),
        "memories": list(a.memories),
        "known": sorted(a.known),
        "relations": rels[:12],
        "requests": [{"from": sim.agents[p["from"]].name, "task": p["task"]} for p in a.pending_requests],
        "history": list(a.history)[-150:],
        "last_prompt": a.last_prompt,
        "last_raw": a.last_raw,
    }


def social_graph(sim: Simulation) -> dict:
    """The relationship graph: who obeys, fears, likes and respects whom."""
    agents = list(sim.agents.values())
    nodes, edges = [], []
    for o in agents:
        others = [x for x in agents if x.id != o.id and x.alive and o.id in x.relations]
        standing = sum(x.relations[o.id].respect + x.relations[o.id].fear for x in others)
        standing += 0.25 * sum(r.obeyed for r in o.relations.values())
        nodes.append({
            "id": o.id, "name": o.name, "color": o.color, "alive": o.alive,
            "ambition": ("changed: " + (o.purpose[:40] + "…" if len(o.purpose) > 40 else o.purpose)) if o.purpose
                        else sim.data["ambitions"][o.ambition].get("label", o.ambition) if o.ambition else None,
            "standing": round(standing, 2),
            "liked": round(sum(x.relations[o.id].affinity for x in others), 2),
            "obeyed_by": sum(1 for r in o.relations.values() if r.obeyed > r.refused),
            "feared_by": sum(1 for x in others if x.relations[o.id].fear > 0.3),
            "group": o.group, "leads": bool(o.group and sim.groups[o.group].leader == o.id),
            "works_for": o.duty["for"] if o.duty else None,
            "crew": sum(1 for x in agents if x.alive and x.duty and x.duty["for"] == o.id),
        })
    for a in agents:
        if not a.alive:
            continue
        for oid, r in a.relations.items():
            if not r.met or not sim.agents[oid].alive:
                continue
            # a's view of oid, plus how often oid obeyed a.
            e = {"from": a.id, "to": oid, "affinity": round(r.affinity, 2), "trust": round(r.trust, 2),
                 "fear": round(r.fear, 2), "respect": round(r.respect, 2), "obeyed": r.obeyed, "refused": r.refused}
            if abs(r.affinity) > 0.25 or r.fear > 0.25 or r.respect > 0.25 or r.obeyed or r.refused:
                edges.append(e)
    groups = [{"name": g.name, "leader": g.leader, "members": sorted(g.members)} for g in sim.groups.values()]
    return {"type": "social", "tick": sim.tick, "nodes": nodes, "edges": edges, "groups": groups}
