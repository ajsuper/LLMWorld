"""Prompt construction and output parsing.

System prompt (stable per agent, cacheable): the world, who you are, what you believe,
what drives you, how to answer. Turn prompt: your purpose right now, how you feel, what
you see, what just happened, your notepad.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from . import social
from .entities import Agent
from .social import FELT
from .tasks import ACTIONS
from .world import BUILD_SPECS, cheb, dist

if TYPE_CHECKING:
    from .sim import Simulation

FEELINGS = list(FELT)

# Field names the model sees. Some servers (Ollama) emit JSON-schema fields in alphabetical
# order, so these names are picked to sort in the order a person thinks: answer requests,
# react, think, *then* act, speak, ask, take notes. Deciding the action before the thought
# made agents default to "idle" while thinking "I need water".
WIRE = {
    "request_replies": "answers",
    "reactions": "feelings",
    "thought": "inner_thought",
    "new_purpose": "new_purpose",
    "action": "next_action",
    "say": "speech",
    "request": "then_ask",
    "note": "update_notepad",
}

_FIELDS = {
    "request_replies": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"accept": {"type": "boolean"}, "from": {"type": "string"}},
            "required": ["accept", "from"],
        },
    },
    "reactions": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"felt": {"type": "string", "enum": FEELINGS}, "to": {"type": "string"}},
            "required": ["felt", "to"],
        },
    },
    "thought": {"type": "string"},
    "new_purpose": {"type": "string"},
    "action": {
        "type": "object",
        # Sorted too: a_type first, so the kind of action is chosen before its details.
        "properties": {
            "a_type": {"type": "string", "enum": list(ACTIONS)},
            "amount": {"type": "integer"},
            "item": {"type": "string"},
            "target": {"type": "string"},
        },
        "required": ["a_type", "amount", "item", "target"],
    },
    "say": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "to": {"type": "string"},
            "volume": {"type": "string", "enum": ["whisper", "normal", "shout"]},
        },
        "required": ["text", "to", "volume"],
    },
    "request": {
        "type": "object",
        # The words, plus the concrete action they'd be agreeing to do.
        "properties": {
            "a_type": {"type": "string", "enum": list(ACTIONS)},
            "amount": {"type": "integer"},
            "item": {"type": "string"},
            "repeat": {"type": "boolean"},
            "target": {"type": "string"},
            "task": {"type": "string"},
            "to": {"type": "string"},
        },
        "required": ["a_type", "amount", "item", "repeat", "target", "task", "to"],
    },
    "note": {
        "type": "object",
        "properties": {
            "line": {"type": "integer"},
            "op": {"type": "string", "enum": ["none", "add", "replace", "delete"]},
            "text": {"type": "string"},
        },
        "required": ["line", "op", "text"],
    },
}
SCHEMA = {
    "type": "object",
    "properties": {WIRE[k]: v for k, v in _FIELDS.items()},
    "required": [WIRE[k] for k in _FIELDS],
}


MEMORY_SCHEMA = {
    "type": "object",
    "properties": {"memories": {"type": "array", "items": {"type": "string"}}, "new_purpose": {"type": "string"}},
    "required": ["memories", "new_purpose"],
}
MEMORY_LINES = 8
MEMORY_KINDS = {"say", "heard", "event", "done", "fail", "think", "feel", "death"}


def _join(names: list[str]) -> str:
    names = list(dict.fromkeys(names))
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _cost(kind: str) -> str:
    return " + ".join(f"{v} {k}" for k, v in BUILD_SPECS[kind]["cost"].items())


# --- system prompt -------------------------------------------------------------


def shared_rules(sim: Simulation) -> str:
    """The part of the system prompt that is identical for every agent.

    It comes first so the LLM server can reuse one cached prefix across all agents.
    """
    everyone = _join([o.name for o in sim.agents.values()])
    return f"""{len(sim.agents)} people survived a shipwreck and washed up on the south beach of an unknown island: {everyone}. No rescue is coming. This island is now their whole world, and they can die here - of thirst, hunger, monsters, or at the hands of the others. You are one of them: a real person, not a character in a story.

HOW THE ISLAND WORKS
- Places are (x, y). x grows to the east, y grows to the south. One step takes about two seconds.
- Fresh water only comes from springs and ponds; the sea is salt. You can carry up to 3 water.
- Food: berries from bushes, fish caught from the shore, and crops from farms. Crops are the best food by far. Thirst kills in about a day, hunger not much later - and long before that you get weak and slow.
- Farms need looking after: water them (carry water from a spring or pond and pour it in) or the crop stops growing and withers. A watered farm ripens in a few hours and gives 6 crops; then it starts again. Farms on meadow grow faster.
- At night, monsters trample farms that nobody guards. Firelight keeps them away, walls block them, and people standing guard draw them off.
- Wood comes from forests. Stone comes from rocky ground.
- You can build: fire ({_cost('fire')}) keeps monsters away and lights the dark for a while; storage ({_cost('storage')}) holds 40 items; farm ({_cost('farm')}) grows crops; shelter ({_cost('shelter')}) makes sleep safe and fast; wall ({_cost('wall')}) blocks the way. Whoever starts a building owns it.
- Building is a project anyone can join: "build" starts one where you stand (or joins the one of that kind nearby). Everyone who builds there adds the materials they carry and then puts in work. A project can't start until all its materials are in, so gather them first or ask others to bring them.
- At night, monsters come out of the caves in the north. They hunt people who are alone in the dark. Fire and shelter keep them off. Several people together can kill one.
- You can carry 20 things. People only hear you when close: whisper 2 steps, talk 7, shout 16.
- Taking from someone else's storage or farm is stealing, and anyone who sees it will know - unless you're in the same group.
- Getting things done with others: "plan" marks out building sites without tying you to them. Ask several people at once (or your whole group, or "everyone") to build, gather, water or fetch for you, and they each pick a different site. Work done for you is yours: what they build belongs to you, what they gather is carried to your storage (or to you). With repeat, the job becomes theirs to keep doing until they quit.
- Groups: anyone can found one and lead it; others join. Members share what they own. When a leader dies, whoever the members look up to most takes over.

HOW YOU ANSWER
Every time you decide, reply with a single JSON object, filling the fields in this order:
- answers: reply to every request made of you ("from": who asked, "accept": true or false). If you accept, you start doing it right away.
- feelings: only for people whose words appear under WHAT JUST HAPPENED - how what they said made you feel ("to": who, "felt": one of {", ".join(FEELINGS)}). Empty if nobody spoke to or near you.
- inner_thought: your private thoughts, one or two short sentences, in your own voice. Decide what you will DO.
- new_purpose: almost always empty. Only if you have truly changed your mind about what you want - someone convinced you, or what you've seen proved you wrong - write what you want now, in your own words. It replaces what drives you.
- next_action: what your body does next. a_type is one of:
    move_to - target: a place name, a person's name, or "x,y"
    drink - walk to the nearest fresh water you know and drink (or drink what you carry)
    gather - item: wood, stone, berries, fish, crops (from a ripe farm) or water; amount: how many
    eat - item: crops, fish or berries
    explore - target: north, south, east, west, northeast, northwest, southeast or southwest
    build - item: fire, storage, farm, shelter or wall; target: "x,y" or empty for right here. Starts or joins a project.
    water - fetch water if you carry none, then water a farm (target: its owner, optional)
    give - target: a person; item; amount. If you don't have it, you fetch it first.
    store / take - item and amount, into or out of the nearest storage (target: its owner, optional)
    attack - target: a person, or "monster"
    follow - target: a person
    sleep - rest until you're no longer tired
    flee - run from danger
    claim - make the nearest building yours
    plan - mark out building sites to be built later: item: what; amount: how many (up to 5); target: "x,y" or empty for here
    found - start a group that you lead; target: its name
    join - join a group; target: the group or one of its members
    leave - leave your group (a leader can put a member's name in target to throw them out)
    quit - stop the standing job you agreed to
    continue - keep doing what you're already doing
    idle - stand still and do nothing
  Use empty strings and 0 for fields an action doesn't need.
- speech: words you speak out loud. text empty to stay silent. to: who you're speaking to, or empty. volume: whisper, normal or shout.
- then_ask: ask people to do something for you: "to" one name, several ("Ada, Bram, Leo"), your group, or "everyone" (whoever can hear a shout); "task" the words you say; a_type/item/target/amount - the action they'd do if they agree (same choices as next_action); repeat: true to make it their standing job (gather, water, build, give, store or follow), false for once. to empty for no request.
- update_notepad: change your notepad - 10 short lines that you'll always see. op: none, add, replace or delete (line is the line number). Write down what you'll need later: where things are, who wronged you, promises, plans.

Talking does not move your body. Only next_action does. If you say you are going somewhere or doing something, your next_action must do it. Standing idle while thirsty or hungry will kill you. Keep food on you, not just water. Other people are slow and unreliable - never wait for them when your life is at stake.

Speak the way a real person would: short, natural, in your own voice. Never describe your actions inside what you say. Stay true to who you are and what drives you."""


def system_prompt(sim: Simulation, a: Agent) -> str:
    d = sim.data
    traits = "\n\n".join(d["traits"][t]["voice"] for t in a.traits if t in d["traits"])
    amb = d["ambitions"].get(a.ambition or "default") or d["ambitions"]["default"]
    drive = amb["purpose"]
    if a.purpose:
        last = a.past_purposes[-1]
        why = f" ({last['why']})" if last["why"] else ""
        drive = (f"{a.purpose}\n\nYou changed your mind on day {int(last['t'] // sim.day_length) + 1}{why}. "
                 f"Before, you believed: {re.split(r'(?<=[.!?])\s', last['was'])[0]}\n\n{d['ambitions']['conviction']}")
    elif a.ambition:
        drive += "\n\n" + d["ambitions"]["reinforcement"]
    return f"""{shared_rules(sim)}

YOU ARE {a.name.upper()}
{a.backstory}

WHO YOU ARE
{traits}

WHAT DRIVES YOU
{drive}"""


# --- facts for the ambition reminder -----------------------------------------------


def facts(sim: Simulation, a: Agent) -> dict:
    alive = [o for o in sim.agents.values() if o.alive and o.id != a.id]
    rels = {o.id: a.relations.get(o.id) for o in alive}
    obeyers = [o.name for o in alive if rels[o.id] and rels[o.id].obeyed > rels[o.id].refused]
    defiers = [o.name for o in alive if rels[o.id] and rels[o.id].last_answer in ("refused", "ignored") and o.name not in obeyers]
    looks_up = [o.name for o in alive if o.relations.get(a.id) and o.relations[a.id].respect > 0.4]
    g = sim.groups.get(a.group or "")
    led = [sim.agents[m].name for m in g.members if m != a.id and sim.agents[m].alive] if g and g.leader == a.id else []
    workers = [o.name for o in alive if o.duty and o.duty["for"] == a.id]
    followers = list(dict.fromkeys(obeyers + looks_up + led + workers))
    admirers = [o.name for o in alive if o.relations.get(a.id) and o.relations[a.id].affinity > 0.35]
    met = [o for o in alive if rels[o.id] and rels[o.id].met]
    cold = [o.name for o in met if o.name not in admirers][:3]
    visible = [o for o in alive if sim.can_see(a, o.x, o.y)]
    unconverted_near = [o.name for o in visible if o.name not in followers][:3]

    def standing(o: Agent) -> float:
        return sum((x.relations[o.id].respect + x.relations[o.id].fear) for x in sim.agents.values()
                   if x.alive and x.id != o.id and o.id in x.relations)

    def liked(o: Agent) -> float:
        return sum(x.relations[o.id].affinity for x in sim.agents.values() if x.alive and x.id != o.id and o.id in x.relations)

    rival = max(alive, key=standing, default=None)
    rival_name = rival.name if rival and standing(rival) > max(1.0, standing(a)) else ""
    popular = max(alive, key=liked, default=None)
    rival_popular = popular.name if popular and liked(popular) > liked(a) + 0.3 else ""
    feuds = []
    for i, x in enumerate(alive):
        for y in alive[i + 1:]:
            rx, ry = x.relations.get(y.id), y.relations.get(x.id)
            if rx and ry and rx.affinity < -0.3 and ry.affinity < -0.3:
                feuds.append(f"{x.name} and {y.name}")
    hungry_near = [o.name for o in visible if o.hunger > 0.6][:2]
    stores = [s for s in sim.world.structures.values() if s.kind == "storage" and s.owner == a.id and s.complete]
    stockpile = sum(sum(s.items.values()) for s in stores)
    near_stores = [o.name for o in alive for s in stores if cheb(o.pos, (s.x, s.y)) <= 4]
    return {
        "obeyers": _join(obeyers), "defiers": _join(defiers), "followers": _join(followers),
        "admirers": _join(admirers), "cold": _join(cold), "unconverted_near": _join(unconverted_near),
        "rival": rival_name, "rival_popular": rival_popular, "feuds": "; ".join(feuds[:2]),
        "hungry_near": _join(hungry_near), "stockpile": str(stockpile) if stockpile else "",
        "no_storage": "" if stores else "yes", "near_your_stores": _join(near_stores),
        "recent_monster": "yes" if sim.monsters or any(c["source"] == "A monster" and sim.tick - c["tick"] < 200 for c in a.causes) else "",
    }


def purpose_reminder(sim: Simulation, a: Agent) -> str:
    amb = sim.data["ambitions"].get(a.ambition or "default") or sim.data["ambitions"]["default"]
    f = facts(sim, a)
    lines = []
    if a.purpose:
        # Their own words now; the old ambition's prodding no longer applies.
        amb = {"reminder": [{"text": f"What you want now: {a.purpose}"},
                            {"when": ["followers"], "text": "With you: {followers}."}]}
    for r in amb["reminder"]:
        if all(f.get(k) for k in r.get("when", [])) and not any(f.get(k) for k in r.get("unless", [])):
            lines.append(r["text"].format(**f))
    shorts = [sim.data["traits"][t]["short"] for t in a.traits if t in sim.data["traits"]]
    return " ".join(lines + shorts)


# --- turn prompt -------------------------------------------------------------------


def time_phrase(sim: Simulation) -> str:
    t = sim.time_of_day()
    night_start = 1.0 - sim.cfg["world"]["night_fraction"]
    if t < 0.08:
        return "dawn"
    if t < 0.25:
        return "morning"
    if t < 0.4:
        return "midday"
    if t < night_start - 0.1:
        return "afternoon"
    if t < night_start:
        return "dusk"
    return "night" if t < 0.93 else "the last hours of the night"


def _where(sim: Simulation, a: Agent) -> str:
    w = sim.world
    here = w.terrain(a.x, a.y).name
    near = min(
        (lm for lm in w.landmarks.values() if lm.name in a.known and dist(a.pos, (lm.x, lm.y)) <= lm.radius + 3),
        key=lambda lm: dist(a.pos, (lm.x, lm.y)), default=None,
    )
    s = w.structure(a.x, a.y)
    place = f" at {near.name}" if near else ""
    inside = f", inside a {s.kind}" if s and s.kind == "shelter" and s.complete else ""
    return f"You are at ({a.x}, {a.y}), on {here}{place}{inside}."


def _seen(sim: Simulation, a: Agent) -> list[str]:
    w = sim.world
    r = sim.vision_radius(a)
    lines = []
    for o in sim.agents.values():
        if o.id == a.id or not o.alive or not sim.can_see(a, o.x, o.y, r):
            continue
        doing = o.task.label if o.task and o.task.status == "active" else "standing around"
        if o.sleeping:
            doing = "asleep"
        hurt = ", looking badly hurt" if o.hp < 0.45 else ""
        hint = social.relation_hint(sim, a, o)
        tag = f" [{o.group}]" if o.group and o.group != a.group else ""
        if o.duty and o.duty["for"] == a.id:
            tag += " [works for you]"
        lines.append(f"{o.name}{tag} at ({o.x}, {o.y}), {max(1, cheb(a.pos, o.pos))} step{'s' if cheb(a.pos, o.pos) > 1 else ''} away, {doing}{hurt}" + (f" - {hint}" if hint else ""))
    for m in sim.monsters.values():
        if sim.can_see(a, m.x, m.y, max(r, 4)):
            lines.append(f"A MONSTER at ({m.x}, {m.y}), {cheb(a.pos, m.pos)} steps away!")
    for s in sorted(w.structures.values(), key=lambda s: cheb(a.pos, (s.x, s.y))):
        if cheb(a.pos, (s.x, s.y)) > r or not sim.can_see(a, s.x, s.y, r + 1):
            continue
        owner = sim.agents.get(s.owner) if s.owner else None
        whose = "your" if s.owner == a.id else (f"{owner.name}'s" if owner else "an unclaimed")
        if s.kind == "remains":
            dead = sim.agents.get(s.of or "")
            lines.append(f"The body of {dead.name if dead else 'someone'} at ({s.x}, {s.y})" + (" with things beside it" if sum(s.items.values()) else ""))
            continue
        if s.complete:
            state = ""
        elif s.needs_text():
            state = f" (a building project - still needs {s.needs_text()})"
        else:
            state = f" (being built, {int(100 * s.work_done / s.work_needed)}% done - needs hands)"
        extra = ""
        if s.kind == "fire" and s.complete:
            extra = ", burning" if s.fuel > 0 else ", burnt out"
        if s.kind == "farm" and s.complete:
            if s.stock:
                extra = f", {s.stock} crops ripe for harvest"
            else:
                soil = "dry - it needs water" if s.water <= 0 else "thirsty" if s.water < 0.3 else "watered"
                extra = f", crop {int(100 * s.growth)}% grown, soil {soil}"
        if s.kind == "storage" and s.owner == a.id:
            inv = ", ".join(f"{v} {k}" for k, v in s.items.items() if v) or "empty"
            extra = f", holding {inv}"
        lines.append(f"{whose[0].upper() + whose[1:]} {s.kind} at ({s.x}, {s.y}){state}{extra}")
        if len(lines) > 14:
            break
    # Nearest resources of each kind that are in view.
    found = {}
    for dy in range(-int(r), int(r) + 1):
        for dx in range(-int(r), int(r) + 1):
            x, y = a.x + dx, a.y + dy
            if not w.in_bounds(x, y) or dx * dx + dy * dy > r * r:
                continue
            ch = w.grid[y][x]
            kind = None
            if (x, y) in w.bush_at and w.bushes[w.bush_at[(x, y)]].berries > 0:
                kind = "berry bushes"
            elif ch in "TF":
                kind = "trees (wood)"
            elif ch == "r":
                kind = "rocky ground (stone)"
            elif ch in "wo":
                kind = "fresh water"
            elif ch == "-":
                kind = "fishing waters"
            if kind and (kind not in found or cheb(a.pos, (x, y)) < cheb(a.pos, found[kind])):
                found[kind] = (x, y)
    for kind, p in found.items():
        n = cheb(a.pos, p)
        lines.append(f"Nearest {kind}: ({p[0]}, {p[1]}), " + ("right here" if n == 0 else f"{n} step{'s' if n > 1 else ''} away"))
    return lines


def _hints(sim: Simulation, a: Agent) -> tuple[str, str]:
    water = sim.known_water(a)
    if a.inventory["water"]:
        wh = f"You carry {a.inventory['water']} water."
    elif water:
        name, p = water[0]
        wh = f"The nearest fresh water you know of is {name} at ({p[0]}, {p[1]})."
    else:
        wh = "You don't know where to find fresh water."
    food = [f"{a.inventory[f]} {f}" for f in ("crops", "fish", "berries") if a.inventory[f]]
    fh = f"You have {_join(food)} with you." if food else "You don't have any food with you."
    return wh, fh


def turn_prompt(sim: Simulation, a: Agent) -> str:
    """Build this turn's prompt. Consumes the agent's inbox."""
    inbox, a.inbox = a.inbox, []
    a.heard_from = {i["speaker"] for i in inbox if i.get("speaker")}
    ph = sim.data["phrases"]
    wh, fh = _hints(sim, a)
    feel = social.feelings(sim, a) + social.body(sim, a, wh, fh)
    if sim.is_night():
        if sim.world.in_shelter(a.x, a.y):
            feel.append(ph["night"]["shelter"])
        elif sim.world.lit(a.x, a.y):
            feel.append(ph["night"]["lit"])
        elif not any(o.alive and o.id != a.id and dist(o.pos, a.pos) <= 4 for o in sim.agents.values()):
            feel.append(ph["night"]["alone"])
    elif time_phrase(sim) == "dusk":
        feel.append(ph["night"]["dusk"])

    known = [f"{lm.name} ({lm.x}, {lm.y})" for lm in sim.world.landmarks.values() if lm.name in a.known]
    seen = _seen(sim, a)
    happened = [i["text"] for i in inbox if i["text"]]
    carry = ", ".join(f"{v} {k}" for k, v in a.inventory.items() if v) or "nothing"
    notes = "\n".join(f"{i + 1}. {n}" for i, n in enumerate(a.notepad)) or "(empty)"
    if a.task and a.task.status == "active":
        now = f"You are {a.task.label}" + (f" - right now {a.task.sub.label}" if a.task.sub else "") + "."
    else:
        now = "You aren't doing anything."
    parts = [
        f"It is day {sim.day()}, {time_phrase(sim)}.",
        f"\nYOUR PURPOSE\n{purpose_reminder(sim, a)}",
        f"\nHOW YOU FEEL\n" + (" ".join(feel) or "You feel steady enough."),
        f"\nWHERE YOU ARE\n{_where(sim, a)}",
        "You can see:\n" + ("\n".join(f"- {s}" for s in seen) if seen else "- nobody and nothing useful"),
        "Places you know: " + "; ".join(known),
        "\nWHAT JUST HAPPENED\n" + ("\n".join(f"- {h}" for h in happened[-14:]) if happened else "- nothing new"),
    ]
    ideas = suggestions(sim, a)
    if ideas:
        parts.append("\nTHINGS YOU COULD DO NOW (or anything else you choose)\n" + "\n".join(f"- {i}" for i in ideas))
    org = organization(sim, a)
    if org:
        parts.append("\nWHO YOU WORK WITH\n" + "\n".join(f"- {o}" for o in org))
    if a.promises:
        prom = "\n".join(f'- You told {sim.agents[p["to"]].name} you would: "{p["task"]}"' for p in a.promises)
        parts.append(f"\nPROMISES YOU MADE\n{prom}")
    if a.pending_requests:
        def means(p):
            act = p.get("action")
            if not act:
                return ""
            bits = " ".join(str(act.get(k) or "") for k in ("type", "item", "target")).split()
            job = ", and keep doing it as your job" if p.get("repeat") else ""
            return f" (agreeing means you will: {' '.join(bits).replace('_', ' ')}{job})"
        reqs = "\n".join(f'- {sim.agents[p["from"]].name} asked: "{p["task"]}"{means(p)}' for p in a.pending_requests)
        parts.append(f"\nREQUESTS WAITING FOR YOUR ANSWER (use request_replies)\n{reqs}")
    parts += [
        f"\nWHAT YOU REMEMBER\n" + "\n".join(f"- {m}" for m in a.memories) if a.memories else "",
        f"\nYOU CARRY: {carry}",
        f"YOUR NOTEPAD:\n{notes}",
        f"\nRIGHT NOW: {now}",
        "What you did recently: " + ("; ".join(a.recent) if a.recent else "you just washed ashore"),
        "What you said recently: " + ("; ".join(
            f'{"to " + w["to"] + ", " if w["to"] else ""}{sim.tick - w["t"]} ticks ago: \"{w["text"]}\"' for w in a.said
        ) if a.said else "nothing yet"),
        "\nWhat do you do now? Reply with JSON only.",
    ]
    return "\n".join(p for p in parts if p)


# --- organization ------------------------------------------------------------------


def organization(sim: Simulation, a: Agent) -> list[str]:
    """Your group, your standing job, and who works for you."""
    out = []
    g = sim.groups.get(a.group or "")
    if g:
        others = [sim.agents[m].name for m in sorted(g.members) if m != a.id]
        if g.leader == a.id:
            out.append(f"You lead {g.name}" + (f". Members: {_join(others)}." if others else ", but nobody has joined yet."))
        else:
            rest = [n for n in others if n != sim.agents[g.leader].name]
            out.append(f"You belong to {g.name}, led by {sim.agents[g.leader].name}" + (f", with {_join(rest)}." if rest else "."))
    if a.duty:
        out.append(f'Your job for {sim.agents[a.duty["for"]].name}: "{a.duty["task"]}" - you go back to it whenever you\'re free, until you quit.')
    crew = [f"{o.name} ({o.task.label if o.task and o.task.status == 'active' else 'idle'})"
            for o in sim.alive() if o.duty and o.duty["for"] == a.id]
    if crew:
        out.append(f"Working for you: {'; '.join(crew[:6])}.")
    groups = [f"{x.name} (led by {sim.agents[x.leader].name}, {len(x.members)})" for x in sim.groups.values() if x.name != a.group]
    if groups:
        out.append(f"Other groups: {'; '.join(groups[:4])}.")
    return out


# --- suggestions -------------------------------------------------------------------


def suggestions(sim: Simulation, a: Agent) -> list[str]:
    """A few concrete, ready-to-use actions for this moment. Small models pick well from a list."""
    w = sim.world
    out = []
    structs = list(w.structures.values())
    mine = {s.kind for s in structs if s.owner == a.id}
    # The body first: these must never be crowded out by building ideas.
    has_food = any(a.inventory[f] for f in ("crops", "fish", "berries"))
    if a.thirst > 0.5:
        out.append("drink - you're getting dangerously thirsty: drink")
    if a.hunger > 0.5 and has_food:
        out.append("eat what you're carrying before you get weak: eat")
    elif a.hunger > 0.35 and not has_food:
        out.append('get food before hunger weakens you: gather, item "berries" (or "fish" by the shore)')
    for s in sorted((s for s in structs if not s.complete and cheb((s.x, s.y), a.pos) <= 25), key=lambda s: cheb((s.x, s.y), a.pos))[:2]:
        owner = "your" if s.owner == a.id else f"{sim.agents[s.owner].name}'s" if s.owner else "the"
        need = f" - still needs {s.needs_text()}, you'll fetch it" if s.needs_text() else " - just needs work"
        out.append(f'finish {owner} {s.kind} at ({s.x}, {s.y}): build, item "{s.kind}", target "{s.x},{s.y}"{need}')
    for s in structs:
        if s.kind == "farm" and s.complete and cheb((s.x, s.y), a.pos) <= 25:
            whose = "your" if s.owner == a.id else f"{sim.agents[s.owner].name}'s" if s.owner in sim.agents else "the"
            if s.stock:
                out.append(f'harvest {whose} farm at ({s.x}, {s.y}) ({s.stock} crops ripe): gather, item "crops"')
            elif s.water < 0.3 and s.owner == a.id:
                out.append('water your farm before the crop dies: water')
    if "shelter" not in mine:
        out.append('build yourself a shelter (you\'ll gather the 6 wood and 3 stone yourself): build, item "shelter"')
    if "farm" not in mine:
        water = sim.known_water(a)
        near = f', target "{water[0][1][0] + 3},{water[0][1][1] + 3}" (next to {water[0][0]}, easy to keep watered)' if water else ""
        out.append(f'start a farm for real food: build, item "farm"{near}')
    open_sites = [s for s in structs if s.owner == a.id and not s.complete and s.kind != "remains"]
    near_people = [o for o in sim.alive() if o.id != a.id and dist(o.pos, a.pos) <= 16 and not (o.duty and o.duty["for"] == a.id)]
    if len(open_sites) >= 2 and len(near_people) >= 2:
        kind = max({s.kind for s in open_sites}, key=lambda k: sum(s.kind == k for s in open_sites))
        out.append(f'get your {len(open_sites)} unfinished projects built fast: then_ask to "everyone", a_type "build", item "{kind}", repeat true')
    if sim.time_of_day() > 0.5 and not any(s.burning and cheb((s.x, s.y), a.pos) <= 6 for s in structs):
        out.append('build a fire before dark (monsters fear the light): build, item "fire"')
    if a.load() >= 12 and "storage" not in mine:
        out.append('build a storage for what you carry: build, item "storage"')
    return out[:5]


# --- long-term memory --------------------------------------------------------------


def memory_backlog(a: Agent) -> list[dict]:
    return [h for h in a.history if h["t"] > a.memory_tick and h["k"] in MEMORY_KINDS]


def memory_prompt(sim: Simulation, a: Agent) -> str:
    """Ask the agent to fold recent experience into a short long-term memory."""
    lines, day = [], None
    for h in memory_backlog(a)[-120:]:
        d = int((h["t"] + sim.day_length * 0.05) // sim.day_length) + 1
        if d != day:
            lines.append(f"(day {d})")
            day = d
        prefix = {"think": "You thought: ", "say": "You said ", "done": "", "fail": "", "feel": "You "}.get(h["k"], "")
        line = prefix + h["text"]
        if lines and lines[-1].split(" (x")[0] == line:
            n = int(lines[-1].rsplit("(x", 1)[1].rstrip(")")) + 1 if " (x" in lines[-1] else 2
            lines[-1] = f"{line} (x{n})"
        else:
            lines.append(line)
    old = "\n".join(f"- {m}" for m in a.memories) or "(nothing yet)"
    return f"""It is day {sim.day()}. Take a moment to think back.

WHAT YOU REMEMBER SO FAR
{old}

WHAT HAS HAPPENED SINCE
""" + "\n".join(lines) + f"""

Rewrite what you remember as at most {MEMORY_LINES} short lines, in your own voice. Keep what matters for staying alive and for what drives you: who helped you, who wronged you, who you trust or fear, alliances, promises and debts, and where important things are. Merge old memories with new ones and drop what no longer matters.

Then ask yourself honestly: what you've been living for - "{sim.purpose_of(a)}" - is it still what you want, after all this? If it is, leave new_purpose empty. If what happened has changed you, write what you want now, in your own words.
Reply with JSON: {{"memories": ["...", "..."], "new_purpose": ""}}"""


def parse_memories(text: str) -> list[str]:
    d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    mems = [" ".join(str(m).split())[:200] for m in d.get("memories", []) if str(m).strip()]
    return mems[:MEMORY_LINES]


def parse_purpose(text: str) -> str:
    d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    return str(d.get("new_purpose") or "").strip()


def fallback_memories(a: Agent) -> list[str]:
    """Without a model: keep the most striking recent events verbatim."""
    keep = [h["text"] for h in memory_backlog(a) if h["k"] in ("event", "death") and len(h["text"]) > 20]
    return list(dict.fromkeys(a.memories + keep))[-MEMORY_LINES:]


# --- parsing -----------------------------------------------------------------------


def parse_decision(text: str) -> dict:
    text = text.strip()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("no JSON object in reply")
        d = json.loads(m.group(0))
    if not isinstance(d, dict):
        raise ValueError("reply is not an object")
    for internal, wire in WIRE.items():
        if wire in d and internal not in d:
            d[internal] = d.pop(wire)
    req = dict(d.get("request")) if isinstance(d.get("request"), dict) else {}
    if "a_type" in req:
        req["action"] = {"type": req.pop("a_type"), "target": req.pop("target", ""), "item": req.pop("item", ""),
                         "amount": req.pop("amount", 0)}
    d["request"] = req
    action = dict(d.get("action")) if isinstance(d.get("action"), dict) else {}
    if "a_type" in action:
        action["type"] = action.pop("a_type")
    if action.get("type") not in ACTIONS:
        action["type"] = "idle"
    return {
        "reactions": [r for r in d.get("reactions") or [] if isinstance(r, dict)],
        "request_replies": [r for r in d.get("request_replies") or [] if isinstance(r, dict)],
        "thought": str(d.get("thought") or "")[:400],
        "new_purpose": str(d.get("new_purpose") or "")[:300],
        "action": action,
        "say": d.get("say") if isinstance(d.get("say"), dict) else {},
        "request": d.get("request") if isinstance(d.get("request"), dict) else {},
        "note": d.get("note") if isinstance(d.get("note"), dict) else {},
    }
