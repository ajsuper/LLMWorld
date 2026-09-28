"""Agent brains: a priority queue of agents waiting to think, served by N parallel workers.

The sim never waits on this. Workers build the prompt from the freshest state when an
agent reaches the front of the queue, call the model, and hand the decision back to be
applied on the next tick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
from typing import TYPE_CHECKING

import aiohttp

from . import prompts
from .world import cheb

if TYPE_CHECKING:
    from .entities import Agent
    from .sim import Simulation

log = logging.getLogger("llmworld.brain")

MEMORY_JOB = "memory:"
MEMORY_EVERY = 30  # new history entries before an agent consolidates memory
MEMORY_URGENT = 90
AGING = 8  # ticks of waiting that are worth one priority level, so nobody starves


class Brain:
    def __init__(self, sim: Simulation, cfg: dict):
        self.sim = sim
        self.cfg = cfg["llm"]
        self.backend = self.cfg["backend"]
        # Jobs waiting for a worker: job id -> (priority, reason). Workers pick the best job at the
        # moment they're free, with priority improving the longer an agent has gone without a
        # thought - otherwise agents in a conversation keep jumping ahead and the rest never think.
        self.waiting: dict[str, int] = {}
        self._ready = asyncio.Event()
        self._systems: dict[str, str] = {}
        self.session: aiohttp.ClientSession | None = None
        self.inflight = 0
        self.errors = 0
        self.last_error = ""
        self.latencies: deque = deque(maxlen=40)
        self.tokens: deque = deque()  # (time, tokens)
        self.decisions = 0
        self._workers: list[asyncio.Task] = []
        self._rng = random.Random(1)

    async def start(self) -> None:
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.cfg["timeout"]))
        if self.backend != "scripted":
            await self._check()
        n = 3 if self.backend == "scripted" else self.cfg["parallel"]
        self._workers = [asyncio.create_task(self._worker()) for _ in range(n)]

    async def close(self) -> None:
        for w in self._workers:
            w.cancel()
        if self.session:
            await self.session.close()

    async def _check(self) -> None:
        url = self.cfg["url"].rstrip("/")
        try:
            if self.backend == "ollama":
                async with self.session.get(f"{url}/api/tags") as r:
                    names = {m["name"] for m in (await r.json()).get("models", [])}
                for tier, model in self.cfg["models"].items():
                    if model not in names and f"{model}:latest" not in names:
                        log.warning("Ollama has no model %r (tier %s). Run: ollama pull %s", model, tier, model)
            else:
                async with self.session.get(f"{url}/v1/models") as r:
                    log.info("LLM server models: %s", [m.get("id") for m in (await r.json()).get("data", [])])
        except Exception as e:  # noqa: BLE001 - just a startup hint
            log.warning("Couldn't reach LLM server at %s (%s). Agents will fall back to scripted behavior.", url, e)

    # --- scheduling ------------------------------------------------------------

    def schedule(self) -> None:
        for prio, aid, reason in self.sim.think_requests():
            a = self.sim.agents[aid]
            a.queued = True
            a.think_reason = reason
            self.waiting[aid] = prio
        # Long-term memory: fold experience into a few lines once enough has piled up.
        # Lowest priority, unless the backlog gets big enough that it would start losing things.
        for a in self.sim.alive():
            if a.remembering:
                continue
            n = len(prompts.memory_backlog(a))
            if n >= MEMORY_EVERY:
                a.remembering = True
                self.waiting[MEMORY_JOB + a.id] = 2 if n >= MEMORY_URGENT else 5
        if self.waiting:
            self._ready.set()

    def _score(self, job: str) -> float:
        aid = job.removeprefix(MEMORY_JOB)
        waited = self.sim.tick - self.sim.agents[aid].last_think
        if job.startswith(MEMORY_JOB):
            waited //= 2  # memory can wait longer than a living decision
        return self.waiting[job] - waited / AGING

    async def _next_job(self) -> str:
        while not self.waiting:
            self._ready.clear()
            await self._ready.wait()
        job = min(self.waiting, key=self._score)
        del self.waiting[job]
        return job

    async def _worker(self) -> None:
        while True:
            aid = await self._next_job()
            if aid.startswith(MEMORY_JOB):
                await self._remember(self.sim.agents[aid.removeprefix(MEMORY_JOB)])
                continue
            a = self.sim.agents[aid]
            a.queued = False
            if not a.alive:
                continue
            a.thinking = True
            self.inflight += 1
            try:
                decision, raw = await self._think(a)
            finally:
                self.inflight -= 1
            a.last_raw = raw
            self.decisions += 1
            self.sim.pending.append((aid, decision, {}))

    def _system(self, a: Agent) -> str:
        # Cached per agent, and rebuilt when they change their mind.
        key = f"{a.id}:{a.purpose_tick}"
        if key not in self._systems:
            self._systems[key] = prompts.system_prompt(self.sim, a)
        return self._systems[key]

    async def _remember(self, a: Agent) -> None:
        start = self.sim.tick
        self.inflight += 1
        try:
            if not a.alive:
                return
            if self.backend == "scripted":
                a.memories = prompts.fallback_memories(a)
            else:
                system = self._system(a)
                model = self.cfg["models"].get(a.tier) or self.cfg["models"]["main"]
                try:
                    raw, _ = await self._complete(system, prompts.memory_prompt(self.sim, a), model, prompts.MEMORY_SCHEMA)
                    a.memories = prompts.parse_memories(raw)
                    if purpose := prompts.parse_purpose(raw):
                        self.sim.change_purpose(a, purpose, "looking back on what happened")
                except Exception as e:  # noqa: BLE001
                    self.errors += 1
                    self.last_error = f"memory: {type(e).__name__}: {e}"[:300]
                    a.memories = prompts.fallback_memories(a)
            a.memory_tick = start
            a.log(self.sim.tick, "memory", "thought back and updated their memories")
        finally:
            a.remembering = False
            self.inflight -= 1

    async def _think(self, a: Agent) -> tuple[dict, str]:
        system = self._system(a)
        user = prompts.turn_prompt(self.sim, a)
        a.last_prompt = user
        t0 = time.monotonic()
        if self.backend == "scripted":
            await asyncio.sleep(self._rng.uniform(0.3, 1.8))
            d = scripted_decision(self.sim, a, self._rng)
            self.latencies.append(time.monotonic() - t0)
            return d, json.dumps(d, indent=1)
        model = self.cfg["models"].get(a.tier) or self.cfg["models"]["main"]
        try:
            raw, toks = await self._complete(system, user, model)
            d = prompts.parse_decision(raw)
        except Exception as e:  # noqa: BLE001 - keep the world moving whatever the model does
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            log.warning("think failed for %s: %s", a.name, self.last_error)
            d = scripted_decision(self.sim, a, self._rng)
            return d, f"(model error: {self.last_error}; used scripted fallback)\n" + json.dumps(d, indent=1)
        now = time.monotonic()
        self.latencies.append(now - t0)
        self.tokens.append((now, toks))
        return d, raw

    async def _complete(self, system: str, user: str, model: str, schema: dict | None = None) -> tuple[str, int]:
        c = self.cfg
        schema = schema or prompts.SCHEMA
        url = c["url"].rstrip("/")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "ollama":
            body = {
                "model": model, "messages": messages, "stream": False, "format": schema,
                "think": False, "keep_alive": "30m",
                "options": {"temperature": c["temperature"], "num_predict": c["max_tokens"], "num_ctx": c["context"]},
            }
            async with self.session.post(f"{url}/api/chat", json=body) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
                data = await r.json()
            return data["message"]["content"], data.get("eval_count", 0)
        body = {
            "model": model, "messages": messages, "temperature": c["temperature"], "max_tokens": c["max_tokens"],
            "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema}},
        }
        if c.get("disable_thinking", True):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Authorization": f"Bearer {c['api_key']}"} if c.get("api_key") else {}
        async with self.session.post(f"{url}/v1/chat/completions", json=body, headers=headers) as r:
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
            data = await r.json()
        return data["choices"][0]["message"]["content"], (data.get("usage") or {}).get("completion_tokens", 0)

    # --- stats -------------------------------------------------------------------

    def stats(self) -> dict:
        now = time.monotonic()
        while self.tokens and now - self.tokens[0][0] > 30:
            self.tokens.popleft()
        lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        return {
            "backend": self.backend, "queued": len(self.waiting), "inflight": self.inflight,
            "latency": round(lat, 2), "tps": round(sum(t for _, t in self.tokens) / 30, 1),
            "errors": self.errors, "last_error": self.last_error, "decisions": self.decisions,
        }


# --- scripted brain ------------------------------------------------------------------
# Used for testing without a GPU, and as a fallback whenever the model errors.

LINES = [
    "We need to find more water.", "Has anyone been past the ridge?", "Stay close after dark.",
    "I'm going to gather wood.", "Anyone seen the berry bushes?", "We should build a fire before night.",
    "I don't trust this place.", "Let's work together.", "Did you hear that?",
]


def scripted_decision(sim: Simulation, a: Agent, rng: random.Random) -> dict:
    d = {"reactions": [], "request_replies": [], "thought": "", "action": {"type": "continue", "target": "", "item": "", "amount": 0},
         "say": {"text": "", "to": "", "volume": "normal"}, "request": {"to": "", "task": ""}, "note": {"op": "none", "line": 0, "text": ""}}

    def act(kind: str, target: str = "", item: str = "", amount: int = 0, thought: str = "") -> None:
        d["action"] = {"type": kind, "target": target, "item": item, "amount": amount}
        d["thought"] = thought

    for sid in a.heard_from:
        d["reactions"].append({"to": sim.agents[sid].name, "felt": rng.choice(["neutral", "neutral", "respected", "insulted", "helped"])})
    for p in a.pending_requests:
        d["request_replies"].append({"from": sim.agents[p["from"]].name, "accept": rng.random() < 0.6})

    busy = a.task and a.task.status == "active" and a.task.kind != "idle"
    m = sim.nearest_monster(a, 6)
    if m and sim.is_night():
        if a.hp > 0.6 and "aggressive" in a.traits:
            act("attack", "monster", thought="I'm not running from that thing.")
        else:
            act("flee", thought="Get away from it!")
        d["say"] = {"text": "Monster!", "to": "", "volume": "shout"}
    elif a.thirst > 0.55 and not (a.task and a.task.kind == "drink"):
        act("drink", thought="I need water.")
    elif a.hunger > 0.5 and a.space() >= 3 and not (a.task and a.task.kind in ("eat", "gather")):
        if a.inventory["berries"] or a.inventory["fish"] or a.inventory["crops"]:
            act("eat", thought="Time to eat.")
        else:
            act("gather", item="berries", amount=6, thought="I need food.")
    elif a.fatigue > 0.75 and not a.sleeping:
        act("sleep", thought="I can barely stand.")
    elif sim.is_night() and a.inventory["wood"] >= 3 and not any(s.burning and cheb((s.x, s.y), a.pos) <= 5 for s in sim.world.structures.values()):
        act("build", item="fire", thought="A fire will keep them away.")
    elif a.space() < 3 and not busy:
        own_storage = any(s.kind == "storage" and s.owner == a.id and s.complete and sum(s.items.values()) < 40
                          for s in sim.world.structures.values())
        if own_storage:
            act("store", thought="My hands are full. Put this away.")
        elif a.inventory["wood"] >= 4:
            act("build", item="storage", thought="I need somewhere to put all this.")
        else:
            act("give", target=next((o.name for o in sim.agents.values() if o.alive and o.id != a.id and cheb(o.pos, a.pos) <= 8), ""),
                item=max(a.inventory, key=a.inventory.get), amount=5, thought="Too much to carry.")
    elif not busy:
        structs = list(sim.world.structures.values())
        own_storage = any(s.kind == "storage" and s.owner == a.id for s in structs)
        own_farm = any(s.kind == "farm" and s.owner == a.id for s in structs)
        near = [s for s in structs if cheb((s.x, s.y), a.pos) <= 10]
        ripe = [s for s in near if s.kind == "farm" and s.complete and s.stock]
        dry = [s for s in near if s.kind == "farm" and s.complete and s.water < 0.3 and not s.stock]
        sites = [s for s in near if not s.complete and any(a.inventory[k] for k, v in s.needs.items() if v > 0)]
        ready = [s for s in near if not s.complete and not s.needs_text()]
        if ripe and rng.random() < 0.7:
            return _finish(d, act("gather", item="crops", amount=6, thought="The crops are ripe."), sim, a, rng)
        if dry and rng.random() < 0.6:
            return _finish(d, act("water", thought="That farm is drying out."), sim, a, rng)
        if (sites or ready) and rng.random() < 0.7:
            s0 = (sites or ready)[0]
            return _finish(d, act("build", item=s0.kind, target=f"{s0.x},{s0.y}", thought=f"I'll help with that {s0.kind}."), sim, a, rng)
        options = [
            lambda: act("gather", item="wood", amount=6, thought="Wood will be useful."),
            lambda: act("gather", item="berries", amount=5, thought="Some food for later."),
            lambda: act("gather", item="stone", amount=4, thought="Stone for building."),
            lambda: act("explore", target=rng.choice(["north", "south", "east", "west", "northeast", "northwest"]), thought="What's over there?"),
            lambda: act("gather", item="water", amount=3, thought="Carry some water."),
        ]
        if a.inventory["wood"] >= 4 and not own_storage:
            options.append(lambda: act("build", item="storage", thought="I need somewhere to keep things."))
        if own_storage and a.load() > 8:
            options.append(lambda: act("store", thought="Put this away."))
        if a.inventory["wood"] >= 4 and a.inventory["stone"] >= 2:
            options.append(lambda: act("build", item="shelter", thought="A proper shelter."))
        if a.inventory["wood"] >= 2 and a.inventory["berries"] >= 2 and not own_farm:
            options += [lambda: act("build", item="farm", thought="Crops would feed us properly.")] * 3
        rng.choice(options)()
    return _finish(d, None, sim, a, rng)


def _finish(d: dict, _, sim: Simulation, a: Agent, rng: random.Random) -> dict:
    visible = [o for o in sim.agents.values() if o.alive and o.id != a.id and cheb(o.pos, a.pos) <= 6]
    if visible and rng.random() < 0.3 and not d["say"]["text"]:
        d["say"] = {"text": rng.choice(LINES), "to": rng.choice(visible).name, "volume": "normal"}
    near = [o for o in visible if cheb(o.pos, a.pos) <= 5]
    if near and rng.random() < (0.25 if a.ambition else 0.05):
        d["request"] = {"to": rng.choice(near).name, "task": rng.choice(["Bring me some wood.", "Fetch me water.", "Follow me.", "Guard my storage."])}
    for lm in sim.world.landmarks.values():
        if lm.name in a.known and not any(lm.name in n for n in a.notepad) and len(a.notepad) < 10 and rng.random() < 0.3:
            d["note"] = {"op": "add", "line": 0, "text": f"{lm.name} is at ({lm.x}, {lm.y})"}
            break
    return d
