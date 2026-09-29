"""The authoritative world simulation. Ticks on a fixed clock; never waits on an LLM."""

from __future__ import annotations

import difflib
import json
import random
import re
import time
from collections import deque
from pathlib import Path

from . import social, tasks
from .entities import NOTEPAD_LINES, NOTEPAD_WIDTH, Agent, Event, Group, Monster, Task
from .world import FARM_GROW_TICKS, FARM_WATER_TICKS, FARM_WITHER_TICKS, FARM_YIELD, cheb, dist
from .worldgen import generate

# Inbox priorities: lower thinks sooner.
P_URGENT, P_NOTICE, P_IDLE, P_HEARTBEAT = 0, 1, 2, 3

NEED_ALERTS = (("thirst", 0.6), ("thirst", 0.85), ("thirst", 0.95), ("hunger", 0.6), ("hunger", 0.85), ("hunger", 0.95),
               ("fatigue", 0.8))
DIRECT_EVENTS = {"request_answer", "monster_attack", "ripe", "withered"}  # always reach their target
OVERHEARD_GAP = 4
UPKEEP_EXEMPT = {"drink", "eat", "sleep", "flee", "attack", "idle"}
PURPOSE_COOLDOWN = 150  # ticks before someone who changed their mind can change it again
REQUEST_REACH = 16.0  # asking is calling out: it carries as far as a shout
REPEATABLE = {"gather", "water", "build", "give", "store", "follow"}  # jobs that make sense to keep doing
EVERYONE = {"everyone", "everybody", "all", "all of you", "anyone", "anybody", "you all"}
MY_GROUP = {"my group", "group", "the group", "us", "my people", "my crew", "my tribe"}  # ticks since an agent's last thought before overheard talk makes them think again


class Simulation:
    def __init__(self, cfg: dict, data: dict, run_dir: Path | None = None):
        self.cfg = cfg
        self.data = data
        wc = cfg["world"]
        self.rng = random.Random(wc["seed"])
        self.world = generate(wc["size"], wc["seed"])
        self.tick = 0
        self.agents: dict[str, Agent] = {}
        self.groups: dict[str, Group] = {}
        self.monsters: dict[int, Monster] = {}
        self.events: list[Event] = []
        self.feed: deque = deque(maxlen=300)
        self.tick_feed: list[dict] = []
        # One feed line per request, rewritten as answers come in: rid -> {item, head, to, answers, tick}
        self.request_lines: dict[int, dict] = {}
        self._rids = 0
        self.pending: list[tuple[str, dict, dict]] = []  # (agent id, decision, meta) from the brain
        self._mods: dict[str, dict] = {}
        self._spawned_tonight = 0
        self._log = None
        if run_dir:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._log = open(run_dir / "events.jsonl", "a", buffering=1)
        self._spawn_agents()

    # --- setup -----------------------------------------------------------

    def _spawn_agents(self) -> None:
        w = self.world
        sx, sy = w.spawn
        spots = [p for p in ((sx + dx, sy + dy) for dy in range(-4, 3) for dx in range(-6, 7)) if w.passable(*p)]
        self.rng.shuffle(spots)
        start_known = {"Shipwreck Beach", "Central Spring"}
        for i, spec in enumerate(self.data["agents"]):
            x, y = spots[i % len(spots)]
            a = Agent(
                id=spec["name"].lower(), name=spec["name"], color=spec["color"],
                tier="main" if spec.get("ambition") else "fast",
                traits=list(spec.get("traits", [])), ambition=spec.get("ambition"),
                backstory=spec.get("backstory", ""), x=x, y=y,
            )
            a.tier = spec.get("tier", a.tier)
            a.known |= start_known
            a.thirst = 0.25 + self.rng.random() * 0.15
            a.hunger = 0.15 + self.rng.random() * 0.15
            for e, v in self.trait_mods(a)["baseline"].items():
                a.emotions[e] = v
            a.emotions["fear"] = max(a.emotions["fear"], 0.2)
            a.causes.append({"emotion": "fear", "source": "The shipwreck", "reason": "left you stranded on an unknown island", "tick": 0, "strength": 0.3})
            self.agents[a.id] = a
            a.log(0, "sys", f"Washed ashore at Shipwreck Beach ({x}, {y}).")

    def trait_mods(self, a: Agent) -> dict:
        m = self._mods.get(a.id)
        if m is None:
            m = {"baseline": {}, "sensitivity": {}, "trust_gain": 1.0, "trust_loss": 1.0}
            for t in a.traits:
                tm = self.data["traits"].get(t, {}).get("mods", {})
                for e, v in tm.get("baseline", {}).items():
                    m["baseline"][e] = m["baseline"].get(e, 0.0) + v
                for e, v in tm.get("sensitivity", {}).items():
                    m["sensitivity"][e] = m["sensitivity"].get(e, 1.0) * v
                m["trust_gain"] *= tm.get("trust_gain", 1.0)
                m["trust_loss"] *= tm.get("trust_loss", 1.0)
            self._mods[a.id] = m
        return m

    # --- time ------------------------------------------------------------

    @property
    def heartbeat(self) -> int:
        return self.cfg["sim"]["heartbeat_ticks"]

    @property
    def day_length(self) -> int:
        return self.cfg["world"]["day_length"]

    def time_of_day(self) -> float:
        return ((self.tick + self.day_length * 0.05) % self.day_length) / self.day_length

    def day(self) -> int:
        return int((self.tick + self.day_length * 0.05) // self.day_length) + 1

    def is_night(self) -> bool:
        return self.time_of_day() >= 1.0 - self.cfg["world"]["night_fraction"]

    # --- lookups ---------------------------------------------------------

    def alive(self):
        return (a for a in self.agents.values() if a.alive)

    def find_agent(self, text: str, exclude: str | None = None) -> Agent | None:
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        for a in self.agents.values():
            if a.id != exclude and a.name.lower() in words:
                return a
        return None

    def stand_pos(self, x: int, y: int):
        return (x, y) if self.world.passable(x, y) else self.world.nearest_passable(x, y, 6)

    def resolve_place(self, a: Agent, text: str):
        """Turn a place description into a position or an Agent, or return an error string."""
        t = (text or "").strip().strip("\"'").strip()
        if not t:
            return "you need to say where"
        m = re.search(r"(-?\d+)\s*[, ]\s*(-?\d+)", t)
        if m:
            x = min(self.world.size - 1, max(0, int(m.group(1))))
            y = min(self.world.size - 1, max(0, int(m.group(2))))
            p = self.stand_pos(x, y)
            return p or f"({x}, {y}) can't be reached on foot"
        other = self.find_agent(t, exclude=a.id)
        if other:
            return other if other.alive else f"{other.name} is dead"
        low = t.lower()
        for lm in self.world.landmarks.values():
            name = lm.name.lower()
            if name in low or (len(low) >= 4 and low.removeprefix("the ") in name):
                if lm.name not in a.known:
                    return f"you don't know where {lm.name} is"
                return self.stand_pos(lm.x, lm.y) or f"{lm.name} can't be reached"
        for kind in ("shelter", "storage", "fire", "farm", "home"):
            if kind in low:
                kind = "shelter" if kind == "home" else kind
                cands = [s for s in self.world.structures.values() if s.kind == kind and s.complete]
                own = [s for s in cands if s.owner == a.id]
                s = min(own or cands, key=lambda s: cheb((s.x, s.y), a.pos), default=None)
                if s:
                    return (s.x, s.y)
                return f"you don't know of any {kind}"
        if "water" in low:
            known = self.known_water(a)
            if known:
                return known[0][1]
        return f"you don't know a place called {t!r}"

    def known_water(self, a: Agent) -> list[tuple[str, tuple[int, int]]]:
        out = []
        for lm in self.world.landmarks.values():
            if lm.kind == "water" and lm.name in a.known:
                out.append((lm.name, (lm.x, lm.y)))
        return sorted(out, key=lambda it: dist(a.pos, it[1]))

    def speed(self, a: Agent) -> float:
        s = 1.0
        if a.fatigue > 0.9:
            s = 0.7
        if a.task and a.task.kind == "flee":
            s += 0.35
        if a.hp < 0.3:
            s *= 0.8
        if max(a.thirst, a.hunger) >= 0.85:
            s *= 0.75  # weak from thirst or hunger
        return s

    def vision_radius(self, a: Agent) -> float:
        t = self.world.terrain(a.x, a.y)
        r = 8.0
        if t.elevated:
            r += 6
        elif t.char == "T":
            r -= 2
        elif t.char == "F":
            r -= 4
        if self.is_night() and not self.world.lit(a.x, a.y):
            r = max(2.5, r * 0.4)
        return r

    def can_see(self, a: Agent, x: int, y: int, radius: float | None = None) -> bool:
        r = radius if radius is not None else self.vision_radius(a)
        if self.is_night() and self.world.lit(x, y):
            r = max(r, 8.0)
        if dist(a.pos, (x, y)) > r:
            return False
        return self.world.line_of_sight(a.pos, (x, y), self.world.terrain(a.x, a.y).elevated)

    def nearest_monster(self, a: Agent, radius: float):
        ms = [m for m in self.monsters.values() if dist(m.pos, a.pos) <= radius]
        return min(ms, key=lambda m: dist(m.pos, a.pos), default=None)

    def recent_attackers(self, a: Agent) -> list[dict]:
        out = []
        for c in a.causes:
            if c["reason"] == "attacked you" and self.tick - c["tick"] < 30:
                other = self.find_agent(c["source"] or "")
                if other and other.alive:
                    out.append({"source_id": other.id})
        return out

    # --- events ----------------------------------------------------------

    def emit(self, ev: Event) -> None:
        self.events.append(ev)

    def _perceives(self, a: Agent, ev: Event) -> bool:
        if dist(a.pos, (ev.x, ev.y)) > ev.radius:
            return False
        if ev.sound:
            return True
        return self.can_see(a, ev.x, ev.y, radius=ev.radius)

    def _describe(self, ev: Event, a: Agent, role: str) -> str:
        actor = self.agents.get(ev.actor) if ev.actor else None
        target = self.agents.get(ev.target) if ev.target else None
        an = actor.name if actor else "Someone"
        d = ev.data
        if ev.kind == "speech":
            vol = d.get("volume", "normal")
            how = {"whisper": "whispering", "shout": "shouting"}.get(vol, "")
            if role == "target":
                return f'{an} (to you{", " + how if how else ""}): "{d["text"]}"'
            if target:
                return f'{an} (to {target.name}{", " + how if how else ""}): "{d["text"]}"'
            return f'{an}{" shouted" if vol == "shout" else ""}: "{d["text"]}"'
        if ev.kind == "request":
            names = [self.agents[i].name for i in d.get("to_ids", [])]
            if role == "target":
                others = [n for n in names if n != a.name]
                also = f" (and {_names(others)})" if 0 < len(others) <= 4 else " (and others)" if others else ""
                job = " - as a standing job, until you quit" if d.get("repeat") else ""
                return f'{an} asks you{also}: "{d["task"]}"{job}'
            return f'{an} asked {_names(names)}: "{d["task"]}"'
        if ev.kind == "request_answer":
            verb = "agreed to" if d["accept"] else "refused"
            if role == "target":
                return f'{an} {verb} your request: "{d["task"]}"'
            return f"{an} {verb} {target.name}'s request"
        if role == "target":
            return {
                "attack": f"{an} attacked you!",
                "gift": f"{an} gave you {d.get('qty')} {d.get('item')}.",
                "theft": f"{an} stole {d.get('qty')} {d.get('item')} from your {d.get('kind')}!",
                "seize": f"{an} took your {d.get('kind')} for themself!",
                "monster_attack": "A monster attacked you!",
                "helped_build": f"{an} helped you build.",
                "farm_raid": f"A monster is tearing up your farm at ({ev.x}, {ev.y})!",
                "ripe": f"The crops on your farm at ({ev.x}, {ev.y}) are ripe. Harvest them (gather crops).",
                "withered": f"Your crops at ({ev.x}, {ev.y}) withered - the farm went too long without water.",
            }.get(ev.kind, ev.text)
        return ev.text

    def _deliver(self) -> None:
        events, self.events = self.events, []
        for ev in events:
            if ev.public:
                item = {"t": self.tick, "k": ev.kind, "text": ev.text, "x": ev.x, "y": ev.y, "actor": ev.actor}
                if ev.kind == "request":
                    item["id"] = f"req{ev.data['rid']}"
                    self.request_lines[ev.data["rid"]] = {"item": item, "head": ev.text, "to": ev.data["to_ids"],
                                                          "answers": {}, "tick": self.tick}
                self.feed.append(item)
                self.tick_feed.append(item)
            if self._log:
                self._log.write(json.dumps({"t": self.tick, "kind": ev.kind, "actor": ev.actor, "target": ev.target,
                                            "text": ev.text, "data": ev.data}) + "\n")
            for a in list(self.alive()):
                if a.id == ev.actor:
                    continue
                is_target = a.id == ev.target or a.id in ev.data.get("to_ids", ())
                if not ((is_target and ev.kind in DIRECT_EVENTS) or self._perceives(a, ev)):
                    continue
                role = "target" if is_target else "witness"
                social.on_event(self, a, ev, role)
                if actor := self.agents.get(ev.actor or ""):
                    a.rel(actor.id).met = True
                if ev.kind == "death" and ev.target:
                    a.seen_bodies.add(ev.target)
                text = self._describe(ev, a, role)
                if not text:
                    continue
                prio = P_NOTICE
                if is_target and ev.kind in ("speech", "request", "attack", "theft", "seize", "monster_attack", "gift", "farm_raid"):
                    prio = P_URGENT
                elif ev.kind in ("built", "claim", "helped_build") and not is_target:
                    prio = P_HEARTBEAT
                elif ev.kind in ("speech", "request", "request_answer") and not is_target:
                    prio = P_IDLE  # overheard: worth a thought, but after anyone spoken to directly
                # Overheard talk waits for a cooldown, so a crowd chatting doesn't flood the queue.
                gap = OVERHEARD_GAP if prio == P_IDLE and ev.kind in ("speech", "request", "request_answer") else 0
                speaker = ev.actor if ev.kind in ("speech", "request") else None
                a.inbox.append({"text": text, "prio": prio, "kind": ev.kind, "speaker": speaker, "tick": self.tick, "gap": gap})
                del a.inbox[:-25]
                a.log(self.tick, "heard" if ev.kind in ("speech", "request") else "event", text)
                if a.sleeping and prio == P_URGENT:
                    self._wake(a)

    def _wake(self, a: Agent) -> None:
        a.sleeping = False
        if a.task and a.task.kind == "sleep" and a.task.status == "active":
            a.task.fail("you were woken up")

    def notify(self, a: Agent, text: str, prio: int = P_NOTICE, kind: str = "notice") -> None:
        a.inbox.append({"text": text, "prio": prio, "kind": kind, "speaker": None, "tick": self.tick})
        del a.inbox[:-25]
        a.log(self.tick, "event", text)

    # --- combat and death -------------------------------------------------

    def hit_agent(self, a: Agent, other: Agent) -> None:
        dmg = 0.1 + 0.06 * a.emotions["anger"]
        other.hp = max(0.0, other.hp - dmg)
        self._wake(other)
        self.emit(Event(self.tick, "attack", other.x, other.y, actor=a.id, target=other.id,
                        text=f"{a.name} attacked {other.name}", radius=10))
        if other.hp <= 0:
            self.kill(other, f"killed by {a.name}", killer=a)

    def hit_monster(self, a: Agent, m: Monster) -> None:
        helpers = sum(1 for o in self.alive() if o.id != a.id and cheb(o.pos, m.pos) <= 1)
        m.hp -= 0.25 + 0.1 * helpers
        if m.hp <= 0:
            self.monsters.pop(m.id, None)
            social.feel(self, a, "joy", 0.2, None, None)
            social.feel(self, a, "fear", -0.2)
            self.emit(Event(self.tick, "monster_killed", m.x, m.y, actor=a.id, text=f"{a.name} killed a monster", radius=12))

    def same_group(self, a: Agent, b: Agent) -> bool:
        return bool(a.group) and a.group == b.group

    def theft(self, thief: Agent, s, item: str, qty: int) -> None:
        owner = self.agents[s.owner]
        if self.same_group(thief, owner):
            return  # a group shares what it has
        ev = Event(self.tick, "theft", s.x, s.y, actor=thief.id, target=owner.id,
                   data={"item": item, "qty": qty, "kind": s.kind},
                   text=f"{thief.name} took {qty} {item} from {owner.name}'s {s.kind}", radius=8)
        if not self._perceives(owner, ev):
            s.thefts.append((thief.id, item, qty))
        self.emit(ev)

    def kill(self, a: Agent, cause: str, killer: Agent | None = None) -> None:
        if not a.alive:
            return
        a.alive = False
        a.hp = 0.0
        a.death_cause = cause
        a.task = None
        a.sleeping = False
        a.bubble = None
        w = self.world
        spot = a.pos if a.pos not in w.structure_at else (w.search(a.pos, lambda p: p if w.buildable(*p) else None, 6) or (a.pos, a.pos))[1]
        if spot not in w.structure_at:
            body = w.add_structure("remains", spot[0], spot[1], None, 0)
            body.items.update(a.inventory)
            body.of = a.id
        a.inventory.clear()
        a.duty = None
        for o in self.agents.values():
            if o.duty and o.duty["for"] == a.id:
                o.duty = None
                self.notify(o, f"With {a.name} dead, your job for them is over.", P_NOTICE)
        if a.group:
            self._leave_group(a, "died")
        text = f"{killer.name} killed {a.name}" if killer else f"{a.name} died ({cause})"
        a.log(self.tick, "death", f"You died: {cause}.")
        self.emit(Event(self.tick, "death", a.x, a.y, actor=killer.id if killer else None, target=a.id,
                        text=text, radius=10, data={"cause": cause}))

    # --- decisions from the brain -----------------------------------------

    def apply_decision(self, a: Agent, d: dict, meta: dict) -> None:
        a.thinking = False
        a.last_think = self.tick
        a.think_count += 1
        if not a.alive:
            return
        for r in d.get("reactions", []):
            other = self.find_agent(r.get("to", ""), exclude=a.id)
            if other and other.id in a.heard_from:
                social.react(self, a, other.id, r.get("felt", "neutral"))
        obey = None
        for rr in d.get("request_replies", []):
            obey = self._answer(a, rr) or obey
        if d.get("thought"):
            a.log(self.tick, "think", d["thought"])
        if d.get("new_purpose"):
            self.change_purpose(a, d["new_purpose"], d.get("thought", ""))
        self._note(a, d.get("note") or {})
        req = d.get("request") or {}
        if req.get("to") and req.get("task"):
            self._request(a, req)
        say = d.get("say") or {}
        if (say.get("text") or "").strip():
            self.say(a, say["text"], say.get("to", ""), say.get("volume", "normal"))
        action = dict(d.get("action") or {"type": "idle"}, _hint=d.get("thought", ""))
        if obey and action.get("type") in ("idle", "continue", "follow", "move_to", obey.get("type")):
            # They agreed to do it, so they do it (unless they chose something urgent instead).
            action = dict(obey, _hint="")
        result = tasks.make_task(self, a, action)
        kind = action.get("type", "idle")
        if isinstance(result, str):
            self.notify(a, f"You tried to {kind.replace('_', ' ')} but {result}.", P_IDLE, "fail")
            a.recent.append(f"tried to {kind} ({result})")
            if not (a.task and a.task.status == "active"):
                a.task = Task("idle", label="waiting", timer=3, started=self.tick)
            return
        if result is None:
            if a.task and a.task.status == "active":
                a.log(self.tick, "act", f"continues {a.task.label}")
                return
            result = Task("idle", label="waiting and watching", timer=self.heartbeat, started=self.tick)
        if a.sleeping and result.kind != "sleep":
            a.sleeping = False
        a.suspended = None
        a.task = result
        a.sleeping = result.kind == "sleep"
        a.recent.append(result.label)
        a.log(self.tick, "act", result.label)

    def say(self, a: Agent, text: str, to_name: str, volume: str) -> None:
        text = " ".join(text.split())[:280]
        volume = volume if volume in ("whisper", "normal", "shout") else "normal"
        radius = {"whisper": 2.5, "normal": 7.0, "shout": 16.0}[volume]
        to = self.find_agent(to_name, exclude=a.id) if to_name else None
        a.bubble = (text, volume, self.tick + 3)
        a.said.append({"t": self.tick, "to": to.name if to else "", "text": text, "volume": volume})
        a.log(self.tick, "say", f'{"(to " + to.name + ") " if to else ""}"{text}"')
        if to and dist(a.pos, to.pos) > radius:
            self.notify(a, f"{to.name} was too far away to hear you.", P_HEARTBEAT)
        self.emit(Event(self.tick, "speech", a.x, a.y, actor=a.id, target=to.id if to else None,
                        data={"text": text, "volume": volume}, radius=radius, sound=True,
                        text=f'{a.name}{" to " + to.name if to else ""}: "{text}"'))

    def recipients(self, a: Agent, text: str) -> list[Agent]:
        """Who a request is for: one name, several ("Ada, Bram and Leo"), a group, or everyone in earshot."""
        t = " ".join(str(text).lower().split())
        if t in EVERYONE:
            return [o for o in self.alive() if o.id != a.id and dist(o.pos, a.pos) <= REQUEST_REACH]
        g = next((g for g in self.groups.values() if g.name.lower() == t or (t in MY_GROUP and g.name == a.group)), None)
        if g:
            return [self.agents[m] for m in sorted(g.members) if m != a.id]
        out = []
        for part in re.split(r",|&|/|\band\b", str(text)):
            o = self.find_agent(part.strip(), exclude=a.id) if part.strip() else None
            if o and o.alive and o not in out:
                out.append(o)
        return out

    def _request(self, a: Agent, req: dict) -> None:
        task = " ".join(str(req["task"]).split())[:200]
        if not task:
            return
        action = req.get("action") if isinstance(req.get("action"), dict) else None
        if action and action.get("type") in ("idle", "continue", None):
            action = None
        repeat = bool(req.get("repeat")) and action is not None and action.get("type") in REPEATABLE
        asked, far = [], []
        for to in self.recipients(a, req["to"]):
            if dist(a.pos, to.pos) > REQUEST_REACH:
                far.append(to.name)
                continue
            if any(p["from"] == a.id and p["task"].lower() == task.lower() for p in to.pending_requests) or \
                    any(p["to"] == a.id and p["task"].lower() == task.lower() for p in to.promises):
                continue  # already asked, or already promised: don't nag
            to.pending_requests = [p for p in to.pending_requests if p["from"] != a.id]
            to.pending_requests.append({"from": a.id, "task": task, "tick": self.tick, "action": action, "repeat": repeat,
                                        "rid": self._rids + 1})
            a.rel(to.id).asked += 1
            asked.append(to)
        if far:
            self.notify(a, f"{_names(far)} {'is' if len(far) == 1 else 'are'} too far away to hear your request.", P_HEARTBEAT)
        if not asked:
            return
        self._rids += 1
        names = _names([o.name for o in asked])
        a.said.append({"t": self.tick, "to": names, "text": f"(asking) {task}", "volume": "normal"})
        a.log(self.tick, "say", f'(asks {names}) "{task}"')
        self.emit(Event(self.tick, "request", a.x, a.y, actor=a.id, target=asked[0].id if len(asked) == 1 else None,
                        data={"task": task, "to_ids": [o.id for o in asked], "repeat": repeat, "rid": self._rids},
                        radius=REQUEST_REACH, sound=True, text=f'{a.name} asked {names}: "{task}"'))

    def _answer(self, a: Agent, rr: dict) -> dict | None:
        asker = self.find_agent(rr.get("from", ""), exclude=a.id)
        if not asker:
            return None
        req = next((p for p in a.pending_requests if p["from"] == asker.id), None)
        if not req:
            return None
        a.pending_requests.remove(req)
        accept = bool(rr.get("accept"))
        r = asker.rel(a.id)
        r.last_answer = "accepted" if accept else "refused"
        if accept:
            r.obeyed += 1
            social.relate(self, a, asker.id, "respect", 0.03)
            a.promises = [p for p in a.promises if p["to"] != asker.id][-2:]
            a.promises.append({"to": asker.id, "task": req["task"], "tick": self.tick})
        else:
            r.refused += 1
        a.log(self.tick, "act", f"{'accepted' if accept else 'refused'} {asker.name}'s request")
        self._request_answered(req.get("rid"), a, "accepted" if accept else "declined")
        obey = dict(req["action"], _for=asker.id) if accept and req.get("action") else None
        if obey and req.get("repeat"):
            a.duty = {"action": dict(obey, _repeat=True), "for": asker.id, "task": req["task"], "since": self.tick}
            a.duty_wait = 0
            a.log(self.tick, "note", f"now works for {asker.name}: {req['task']}")
        self.emit(Event(self.tick, "request_answer", a.x, a.y, actor=a.id, target=asker.id,
                        data={"accept": accept, "task": req["task"]}, radius=7.0, sound=True, public=False,
                        text=f"{a.name} {'agreed to' if accept else 'refused'} {asker.name}'s request"))
        return obey

    # --- changing one's mind ----------------------------------------------

    def purpose_of(self, a: Agent) -> str:
        if a.purpose:
            return a.purpose
        amb = self.data["ambitions"].get(a.ambition or "default") or self.data["ambitions"]["default"]
        return " ".join(amb["purpose"].split())

    def change_purpose(self, a: Agent, text: str, why: str = "") -> bool:
        """The agent has been convinced: what drives them is now whatever they say it is."""
        text = " ".join(str(text).split()).strip(' "')[:300]
        if len(text) < 12 or not a.alive:
            return False
        was = self.purpose_of(a)
        if difflib.SequenceMatcher(None, text.lower(), was.lower()).ratio() > 0.7:
            return False  # restating what they already want isn't a change of heart
        if self.tick - a.purpose_tick < PURPOSE_COOLDOWN:
            a.log(self.tick, "feel", f"felt pulled toward something new ({text}), but it's too soon to turn around again")
            return False
        a.past_purposes.append({"t": self.tick, "was": was, "now": text, "why": " ".join(str(why).split())[:200]})
        a.purpose, a.purpose_why, a.purpose_tick = text, " ".join(str(why).split())[:200], self.tick
        a.log(self.tick, "memory", f"Changed your mind. What you want now: {text}")
        # Private: nobody perceives it (radius < 0), but the viewer's feed shows it.
        self.emit(Event(self.tick, "change_of_heart", a.x, a.y, actor=a.id, radius=-1,
                        text=f"{a.name} had a change of heart: {text}"))
        return True

    # --- groups and jobs ---------------------------------------------------

    def organize(self, a: Agent, kind: str, target: str) -> tuple[bool, str]:
        """found / join / leave / quit. Returns (ok, what happened)."""
        if kind == "quit":
            if not a.duty:
                return False, "you don't have a standing job"
            boss = self.agents[a.duty["for"]]
            a.duty = None
            self.emit(Event(self.tick, "quit", a.x, a.y, actor=a.id, target=boss.id, radius=7,
                            text=f"{a.name} quit working for {boss.name}"))
            return True, f"you stopped working for {boss.name}"
        if kind == "found":
            name = " ".join(target.replace('"', "").split())[:30].strip(" .") or f"{a.name}'s people"
            if name.lower() in EVERYONE | MY_GROUP or self.find_agent(name):
                return False, f"{name!r} can't be a group's name"
            if any(g.name.lower() == name.lower() for g in self.groups.values()):
                return False, f"there is already a group called {name}"
            if a.group:
                self._leave_group(a, "left")
            self.groups[name] = Group(name, a.id, {a.id}, self.tick)
            a.group = name
            self.emit(Event(self.tick, "group", a.x, a.y, actor=a.id, radius=12,
                            text=f"{a.name} founded a group, {name}, and leads it"))
            return True, f"you founded {name}. Ask people to join it (a_type join, target {name})"
        if kind == "join":
            t = target.strip().lower()
            g = next((g for g in self.groups.values() if g.name.lower() == t), None)
            if not g:
                o = self.find_agent(target, exclude=a.id)
                g = self.groups.get(o.group) if o and o.group else None
            if not g:
                return False, f"there's no group called {target!r}"
            if a.group == g.name:
                return False, f"you're already in {g.name}"
            if a.group:
                self._leave_group(a, "left")
            g.members.add(a.id)
            a.group = g.name
            leader = self.agents[g.leader]
            social.relate(self, a, leader.id, "respect", 0.1)
            self.emit(Event(self.tick, "group", a.x, a.y, actor=a.id, target=leader.id, radius=12,
                            text=f"{a.name} joined {g.name}, led by {leader.name}"))
            return True, f"you joined {g.name}, led by {leader.name}"
        if kind == "leave":
            if not a.group:
                return False, "you're not in a group"
            g = self.groups[a.group]
            if target:
                o = self.find_agent(target, exclude=a.id)
                if not o or o.id not in g.members:
                    return False, f"{target} isn't in {g.name}"
                if g.leader != a.id:
                    return False, f"only {self.agents[g.leader].name} can throw people out of {g.name}"
                self._leave_group(o, "thrown out", by=a)
                return True, f"you threw {o.name} out of {g.name}"
            name = g.name
            self._leave_group(a, "left")
            return True, f"you left {name}"
        return False, f"unknown action {kind!r}"

    def _leave_group(self, a: Agent, how: str, by: Agent | None = None) -> None:
        g = self.groups.get(a.group or "")
        a.group = None
        if not g:
            return
        g.members.discard(a.id)
        if how != "died":
            text = f"{by.name} threw {a.name} out of {g.name}" if by else f"{a.name} left {g.name}"
            self.emit(Event(self.tick, "group", a.x, a.y, actor=by.id if by else a.id, target=a.id if by else None,
                            radius=12, text=text))
        alive = [self.agents[m] for m in g.members if self.agents[m].alive]
        if not alive:
            del self.groups[g.name]
            return
        if g.leader == a.id:
            # Whoever the others look up to most takes over.
            def pull(o):
                return sum(x.rel(o.id).respect + x.rel(o.id).fear for x in alive if x.id != o.id)
            heir = max(alive, key=pull)
            g.leader = heir.id
            ev = Event(self.tick, "group", heir.x, heir.y, actor=heir.id, radius=12,
                       text=f"{heir.name} now leads {g.name}")
            ev.data["to_ids"] = [m.id for m in alive]  # everyone in the group hears of it
            self.emit(ev)

    def _resume_duty(self, a: Agent) -> None:
        """Nothing else going on: the body goes back to the job it agreed to."""
        d = a.duty
        if not d or a.sleeping or self.tick < a.duty_wait or a.suspended:
            return
        if a.task and a.task.status == "active" and a.task.kind != "idle":
            return
        boss = self.agents[d["for"]]
        t = tasks.make_task(self, a, d["action"])
        if not isinstance(t, Task):
            a.duty_wait = self.tick + 60
            self.notify(a, f"There's nothing to do on your job for {boss.name} right now"
                           + (f" ({t})." if isinstance(t, str) else "."), P_IDLE)
            return
        a.task = t
        a.recent.append(f"(job for {boss.name}) {t.label}")
        a.log(self.tick, "act", f"back to work for {boss.name}: {t.label}")

    def _request_answered(self, rid: int | None, a: Agent, how: str) -> None:
        """Fold an answer into the request's feed line: 'Bob asked ...: "..." - Ada accepted, Leo declined'."""
        line = self.request_lines.get(rid)
        if not line:
            return
        line["answers"][a.id] = how
        by: dict[str, list[str]] = {}
        for aid, h in line["answers"].items():
            by.setdefault(h, []).append(self.agents[aid].name)
        order = ("accepted", "declined", "didn't answer")
        parts = [f"{_names(by[h])} {h}" for h in order if h in by]
        line["item"]["text"] = f"{line['head']} - {', '.join(parts)}"
        self.tick_feed.append(line["item"])  # same id: the viewer replaces the old line
        if len(line["answers"]) >= len(line["to"]):
            del self.request_lines[rid]
        for k in [k for k, v in self.request_lines.items() if self.tick - v["tick"] > 200]:
            del self.request_lines[k]  # someone died before answering

    def _note(self, a: Agent, note: dict) -> None:
        op = note.get("op", "none")
        text = " ".join(str(note.get("text", "")).split())[:NOTEPAD_WIDTH]
        try:
            line = int(note.get("line") or 0) - 1
        except (TypeError, ValueError):
            line = -1
        if op == "add" and text:
            if len(a.notepad) >= NOTEPAD_LINES:
                self.notify(a, "Your notepad is full. Replace or delete a line first.", P_HEARTBEAT)
                return
            a.notepad.append(text)
        elif op == "replace" and text and 0 <= line < len(a.notepad):
            a.notepad[line] = text
        elif op == "delete" and 0 <= line < len(a.notepad):
            text = a.notepad.pop(line)
        else:
            return
        a.log(self.tick, "note", f"{op}: {text}")

    # --- the tick ----------------------------------------------------------

    def step(self) -> None:
        self.tick += 1
        self.world.tile_changes.clear()
        self.world.bush_changes.clear()
        self.tick_feed = []
        for aid, d, meta in self.pending:
            self.apply_decision(self.agents[aid], d, meta)
        self.pending = []
        self._update_world()
        self._update_monsters()
        for a in list(self.alive()):
            self._update_agent(a)
        self._deliver()

    def _update_world(self) -> None:
        w = self.world
        for b in w.bushes.values():
            if b.berries < b.max:
                b.regrow += 1
                if b.regrow >= 60:
                    b.regrow = 0
                    b.berries += 1
                    w.bush_changes.add(b.id)
        for s in list(w.structures.values()):
            if s.kind == "fire" and s.complete and s.fuel > 0:
                s.fuel -= 1
                if s.fuel == 0:
                    w.structures_dirty = True
            elif s.kind == "farm" and s.complete and s.stock == 0:
                if s.water > 0:
                    s.water = max(0.0, s.water - 1 / FARM_WATER_TICKS)
                    s.dry = 0
                    rate = 1.5 if w.grid[s.y][s.x] == '"' else 1.0
                    s.growth += rate / FARM_GROW_TICKS
                    if s.growth >= 1.0:
                        s.growth = 0.0
                        s.stock = FARM_YIELD
                        w.structures_dirty = True
                        self.emit(Event(self.tick, "ripe", s.x, s.y, target=s.owner, radius=10,
                                        text=f"The crops on the farm at ({s.x}, {s.y}) are ripe"))
                else:
                    s.dry += 1
                    if s.dry == FARM_WITHER_TICKS and s.growth > 0:
                        s.growth = 0.0
                        w.structures_dirty = True
                        self.emit(Event(self.tick, "withered", s.x, s.y, target=s.owner, radius=10,
                                        text=f"The crops on the farm at ({s.x}, {s.y}) withered for lack of water"))
                if self.tick % 10 == 0:
                    w.structures_dirty = True
            elif s.kind == "remains" and not sum(s.items.values()) and self.tick % 50 == 0:
                s.grow += 50
                if s.grow > 1500:
                    w.remove_structure(s)
        if self.tick % 40 == 0 and w.forest_origin:
            for _ in range(3):
                x, y = self.rng.choice(w.forest_origin)
                if w.grid[y][x] == "," and (x, y) not in w.structure_at:
                    w.set_tile(x, y, "T")
                    w.wood[(x, y)] = 4

    def _update_agent(self, a: Agent) -> None:
        ns = self.cfg["sim"]["needs"]
        night = self.is_night()
        in_shelter = self.world.in_shelter(a.x, a.y)
        a.thirst = min(1.0, a.thirst + 1 / ns["thirst_ticks"])
        a.hunger = min(1.0, a.hunger + 1 / ns["hunger_ticks"])
        if a.sleeping:
            a.fatigue = max(0.0, a.fatigue - (1 / 35 if in_shelter else 1 / 60))
            a.hp = min(1.0, a.hp + (0.006 if in_shelter else 0.003))
        else:
            a.fatigue = min(1.0, a.fatigue + (1.5 if night else 1.0) / ns["fatigue_ticks"])
        # Past 0.9 the body starts to fail; at 1.0 it fails fast.
        a.hp -= 0.012 if a.thirst >= 1.0 else 0.002 if a.thirst >= 0.9 else 0
        a.hp -= 0.009 if a.hunger >= 1.0 else 0.002 if a.hunger >= 0.9 else 0
        if a.thirst < 0.6 and a.hunger < 0.6 and not a.sleeping:
            a.hp = min(1.0, a.hp + 0.0015)
        if a.hp <= 0:
            cause = "of thirst" if a.thirst >= 1 else "of hunger" if a.hunger >= 1 else "of their wounds"
            return self.kill(a, cause)
        if a.fatigue >= 1.0 and not a.sleeping:
            a.task = Task("sleep", label="collapsed from exhaustion", started=self.tick)
            a.sleeping = True
            self.notify(a, "You collapsed from exhaustion.", P_URGENT)

        social.decay(self, a)
        if night and not self.world.lit(a.x, a.y) and not in_shelter:
            company = sum(1 for o in self.alive() if o.id != a.id and dist(o.pos, a.pos) <= 4)
            if not company:
                social.feel(self, a, "fear", 0.004)

        for need, level in NEED_ALERTS:
            flag = f"{need}{level}"
            v = getattr(a, need)
            if v >= level and flag not in a.need_flags:
                a.need_flags.add(flag)
                a.inbox.append({"text": "", "prio": P_NOTICE, "kind": "need", "speaker": None, "tick": self.tick})
            elif v < level - 0.15:
                a.need_flags.discard(flag)

        self._instinct(a)
        self._upkeep(a)
        self._resume_duty(a)
        t = a.task
        if t and t.status == "active":
            tasks.step(self, a, t)
            if t.status != "active" and a.suspended is not None:
                # Upkeep finished: pick the job back up without spending a thought on it.
                a.log(self.tick, "done", t.result)
                a.task, a.suspended = a.suspended, None
                a.task.path = []
                t = a.task
            if t.status != "active":
                if t.kind == "sleep":
                    a.sleeping = False
                if t.status == "failed" and t.repeat:
                    a.duty_wait = self.tick + 40  # the job can't be done right now; don't hammer at it
                ok = t.status == "done"
                a.log(self.tick, "done" if ok else "fail", t.result)
                a.recent.append(f"{t.label}: {t.result}")
                a.inbox.append({"text": f"You finished {t.label}: {t.result}." if ok else f"You stopped {t.label}: {t.result}.",
                                "prio": P_IDLE, "kind": "task", "speaker": None, "tick": self.tick})

        self._discover(a)
        self._notice(a)
        if a.bubble and self.tick > a.bubble[2]:
            a.bubble = None
        a.promises = [p for p in a.promises if self.tick - p["tick"] < 150 and self.agents[p["to"]].alive]
        for p in list(a.pending_requests):
            if self.tick - p["tick"] > 90:
                a.pending_requests.remove(p)
                asker = self.agents.get(p["from"])
                self._request_answered(p.get("rid"), a, "didn't answer")
                if asker:
                    asker.rel(a.id).refused += 1
                    asker.rel(a.id).last_answer = "ignored"

    def _upkeep(self, a: Agent) -> None:
        """Mid-job, the body looks after itself: pause, drink or eat, then get back to it.

        Keeps an agent's few decisions for things that matter instead of water runs.
        """
        t = a.task
        if a.sleeping or not t or t.status != "active" or t.kind in UPKEEP_EXEMPT or a.suspended:
            return
        if a.thirst >= 0.65 and (a.inventory["water"] or self.known_water(a)):
            job = tasks.make_task(self, a, {"type": "drink"})
        elif a.hunger >= 0.65 and not (t.kind == "gather" and t.item in tasks.FOOD):
            job = self._food_task(a)
            if job is None:
                return
        else:
            return
        if isinstance(job, Task):
            job.label = f"{job.label} (then back to {t.label})"
            a.suspended, a.task = t, job
            a.log(self.tick, "act", job.label)

    def _food_task(self, a: Agent) -> Task | None:
        """Eat what you carry, or else go and get the nearest food."""
        if any(a.inventory[f] for f in tasks.FOOD):
            t = tasks.make_task(self, a, {"type": "eat"})
            return t if isinstance(t, Task) else None
        ripe = any(s.kind == "farm" and s.stock for s in self.world.structures.values())
        near = [(dist(a.pos, found[0]), item) for item in (("crops",) if ripe else ()) + ("berries", "fish")
                if (found := tasks._find_source(self, a, item))]
        if not near:
            return None
        t = tasks.make_task(self, a, {"type": "gather", "item": min(near)[1], "amount": 4})
        return t if isinstance(t, Task) else None

    def _instinct(self, a: Agent) -> None:
        """When the body is about to fail and the mind is standing around, the body takes over."""
        if a.sleeping or (a.task and a.task.status == "active" and a.task.kind not in ("idle", "follow")):
            return
        if a.thirst >= 0.85 and (a.inventory["water"] or self.known_water(a)):
            t = tasks.make_task(self, a, {"type": "drink"})
            text = "Your body takes over: you can't stand the thirst and head for water."
        elif a.hunger >= 0.8 and (t := self._food_task(a)) is not None:
            text = ("Your body takes over: you wolf down the food you're carrying." if t.kind == "eat"
                    else f"Your body takes over: you're starving, and you go looking for {t.item}.")
        else:
            return
        if isinstance(t, Task):
            a.task = t
            a.recent.append(f"(instinct) {t.label}")
            self.notify(a, text, P_NOTICE, "instinct")

    def _discover(self, a: Agent) -> None:
        r = self.vision_radius(a)
        for lm in self.world.landmarks.values():
            if lm.name in a.known:
                continue
            d = dist(a.pos, (lm.x, lm.y))
            if lm.kind == "area":
                ok = d <= lm.radius + 3
            elif lm.hidden:
                ok = d <= 4 or (d <= r and self.world.line_of_sight(a.pos, (lm.x, lm.y)))
            else:
                ok = d <= r + lm.radius and self.world.line_of_sight(a.pos, (lm.x, lm.y), self.world.terrain(a.x, a.y).elevated)
            if ok:
                a.known.add(lm.name)
                self.notify(a, f"You discovered {lm.name} at ({lm.x}, {lm.y}).", P_NOTICE, "discover")
                if lm.kind == "water":
                    social.feel(self, a, "joy", 0.15, lm.name, "has fresh water")

    def _notice(self, a: Agent) -> None:
        w = self.world
        r = self.vision_radius(a)
        for s in w.structures.values():
            if s.kind == "remains" and s.of and s.of not in a.seen_bodies and s.of != a.id and self.can_see(a, s.x, s.y, r):
                a.seen_bodies.add(s.of)
                dead = self.agents[s.of]
                social.on_event(self, a, Event(self.tick, "death", s.x, s.y, target=dead.id), "witness")
                self.notify(a, f"You found the body of {dead.name} at ({s.x}, {s.y}). {dead.name} died {dead.death_cause}.", P_URGENT)
            elif s.owner == a.id and s.thefts and cheb((s.x, s.y), a.pos) <= 2:
                lost = {}
                for _, item, qty in s.thefts:
                    lost[item] = lost.get(item, 0) + qty
                s.thefts.clear()
                what = ", ".join(f"{q} {i}" for i, q in lost.items())
                social.feel(self, a, "anger", 0.2, "Someone", f"stole {what} from your {s.kind}")
                self.notify(a, f"Someone took {what} from your {s.kind} while you were away.", P_URGENT)
        if self.monsters and self.tick - a.last_monster_alert > 15:
            m = self.nearest_monster(a, r + 2)
            if m and self.can_see(a, m.x, m.y, max(r, 4)):
                a.last_monster_alert = self.tick
                social.feel(self, a, "fear", 0.12, "A monster", "is prowling nearby")
                self.notify(a, f"You see a monster at ({m.x}, {m.y})!", P_NOTICE, "monster")

    # --- monsters ------------------------------------------------------------

    def _update_monsters(self) -> None:
        mc = self.cfg["monsters"]
        w = self.world
        night = self.is_night()
        if night:
            want = min(mc["base"] + (self.day() - 1) * mc["per_day"], mc["max"])
            if self._spawned_tonight < want and self.tick % 5 == 0:
                spot = w.nearest_passable(*w.cave, 2)
                if spot:
                    m = Monster(w.new_id(), *spot)
                    self.monsters[m.id] = m
                    self._spawned_tonight += 1
        else:
            self._spawned_tonight = 0
        for m in list(self.monsters.values()):
            if m.cooldown:
                m.cooldown -= 1
            if not night:
                if cheb(m.pos, w.cave) <= 1:
                    del self.monsters[m.id]
                    continue
                self._monster_walk(m, w.cave)
                continue
            prey = None
            best = 11.0
            for a in self.alive():
                d = dist(a.pos, m.pos)
                if d < best and not w.in_shelter(a.x, a.y) and not w.lit(a.x, a.y) and w.line_of_sight(m.pos, a.pos):
                    prey, best = a, d
            if prey:
                if cheb(prey.pos, m.pos) <= 1:
                    if not m.cooldown:
                        m.cooldown = 2
                        prey.hp = max(0.0, prey.hp - 0.07)
                        self._wake(prey)
                        self.emit(Event(self.tick, "monster_attack", prey.x, prey.y, target=prey.id, radius=12, sound=True,
                                        text=f"A monster attacked {prey.name}"))
                        if prey.hp <= 0:
                            self.kill(prey, "killed by a monster")
                    continue
                if m.target != prey.id or m.repath <= 0:
                    m.path = w.find_path(m.pos, prey.pos, near=1, max_expand=2500) or []
                    m.target, m.repath = prey.id, 3
                m.repath -= 1
            elif self._raid_farm(m):
                continue
            elif not m.path:
                m.target = None
                for _ in range(8):
                    gx = m.x + self.rng.randint(-10, 10)
                    gy = m.y + self.rng.randint(-10, 10)
                    if w.passable(gx, gy) and dist((gx, gy), w.cave) < 40 and not w.lit(gx, gy):
                        m.path = w.find_path(m.pos, (gx, gy), max_expand=1500) or []
                        break
            self._monster_step(m)

    def _raid_farm(self, m: Monster) -> bool:
        """With no one to hunt, monsters go for crops. Firelight and walls keep them off."""
        w = self.world
        farms = [s for s in w.structures.values() if s.kind == "farm" and s.complete and (s.stock or s.growth > 0.05)
                 and dist((s.x, s.y), m.pos) < 30 and not w.lit(s.x, s.y)]
        farm = min(farms, key=lambda s: dist((s.x, s.y), m.pos), default=None)
        if not farm:
            return False
        if cheb(m.pos, (farm.x, farm.y)) <= 1:
            if not m.cooldown:
                m.cooldown = 3
                if farm.stock:
                    farm.stock -= 1
                else:
                    farm.growth = max(0.0, farm.growth - 0.08)
                w.structures_dirty = True
                owner = self.agents.get(farm.owner or "")
                if self.tick - farm.grow < 12:  # farm.grow doubles as "last raid announced" for farms
                    return True
                farm.grow = self.tick
                self.emit(Event(self.tick, "farm_raid", farm.x, farm.y, target=farm.owner, radius=16, sound=True,
                                text=f"A monster is tearing up {owner.name + chr(39) + 's' if owner else 'a'} farm at ({farm.x}, {farm.y})"))
            return True
        if m.target != f"farm{farm.id}" or not m.path:
            m.path = w.find_path(m.pos, (farm.x, farm.y), near=1, max_expand=2500) or []
            m.target = f"farm{farm.id}"
        self._monster_step(m)
        return True

    def _monster_walk(self, m: Monster, goal) -> None:
        if not m.path:
            m.path = self.world.find_path(m.pos, goal, near=1, max_expand=4000) or []
        self._monster_step(m)

    def _monster_step(self, m: Monster) -> None:
        if not m.path:
            return
        w = self.world
        nxt = m.path[0]
        if not w.passable(*nxt) or (self.is_night() and w.lit(*nxt)):
            m.path = []
            return
        cost = w.step_cost(m.pos, nxt)
        m.move_points = min(m.move_points + 0.85, max(2.0, cost))
        if m.move_points >= cost:
            m.move_points -= cost
            m.x, m.y = nxt
            m.path.pop(0)

    # --- scheduling ------------------------------------------------------------

    def think_requests(self) -> list[tuple[int, str, str]]:
        out = []
        for a in self.alive():
            if a.thinking or a.queued:
                continue
            prio, reason = None, ""
            since = self.tick - a.last_think
            ready = [i for i in a.inbox if since >= i.get("gap", 0)]
            if ready:
                top = min(ready, key=lambda i: i["prio"])
                prio, reason = top["prio"], top["text"] or top["kind"]
            idle = a.task is None or a.task.status != "active"
            if idle and (prio is None or prio > P_IDLE):
                prio, reason = P_IDLE, "nothing to do"
            if prio is None and self.tick - a.last_think >= self.heartbeat and not a.sleeping:
                prio, reason = P_HEARTBEAT, "checking in"
            if prio is None:
                continue
            if a.sleeping and prio > P_URGENT:
                continue
            if prio >= P_HEARTBEAT and self.tick - a.last_think < 3:
                continue
            out.append((prio, a.id, reason))
        return out

    def close(self) -> None:
        if self._log:
            self._log.close()


def _names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def new_run_dir(root: Path) -> Path:
    return root / time.strftime("%Y%m%d-%H%M%S")
