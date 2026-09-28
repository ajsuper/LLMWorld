"""Actions as multi-tick tasks.

The LLM picks *intent* ("gather wood", "move_to Central Spring"); the sim handles
pathfinding, finding the nearest source, and doing the work tick by tick, so an
agent keeps busy while its next thought is being generated.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from .entities import CAPACITY, ITEMS, WATER_CAPACITY, Agent, Event, Task
from .world import BUILD_SPECS, DIRS8, STORAGE_CAPACITY, WATER_PER_UNIT, cheb

if TYPE_CHECKING:
    from .sim import Simulation

ACTIONS = (
    "continue", "idle", "move_to", "explore", "gather", "drink", "eat", "build",
    "give", "store", "take", "attack", "follow", "sleep", "flee", "claim", "water",
    "plan", "found", "join", "leave", "quit",
)
ORGANIZE = ("found", "join", "leave", "quit")  # instant: they change who you belong to, not where you are
MAX_OPEN_SITES = 8
FOOD = {"crops": 0.45, "fish": 0.4, "berries": 0.2}
GATHER_TICKS = {"wood": 2, "stone": 3, "berries": 1, "fish": 3, "crops": 1, "water": 1}
DIRECTIONS = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
    "northeast": (1, -1), "northwest": (-1, -1), "southeast": (1, 1), "southwest": (-1, 1),
}


def _item(text: str) -> str:
    t = text.strip().lower()
    for it in ITEMS:
        if t.startswith(it[:4]):
            return it
    if t in ("berry", "food"):
        return "berries"
    if t.startswith(("crop", "vegetable", "grain", "harvest")):
        return "crops"
    return t


def make_task(sim: Simulation, a: Agent, action: dict) -> Task | str | None:
    """Build a task from an LLM action. Returns a Task, an error string, or None (keep current)."""
    kind = action.get("type", "idle")
    target = str(action.get("target", "") or "").strip()
    item = _item(str(action.get("item", "") or ""))
    try:
        amount = int(action.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0
    t = Task(kind, target=target, item=item, amount=amount, started=sim.tick)
    boss = sim.agents.get(action.get("_for") or "")
    if boss and boss.alive and boss.id != a.id:
        # Doing it for someone who asked: an empty or vague target means them.
        t.for_id = boss.id
        t.repeat = bool(action.get("_repeat"))
        if kind in ("give", "follow", "move_to") and not sim.find_agent(target, exclude=a.id) \
                and not (kind == "move_to" and target):
            t.target = target = boss.name

    if kind == "continue":
        return None
    if kind == "idle":
        t.timer = sim.heartbeat
        t.label = "waiting and watching"
        return t
    if kind in ("move_to", "follow"):
        dest = sim.resolve_place(a, target)
        if isinstance(dest, str):
            return dest
        if isinstance(dest, Agent):
            if dest.id == a.id:
                return "you can't go to yourself"
            t.target_id = dest.id
            t.label = f"{'following' if kind == 'follow' else 'going to'} {dest.name}"
        else:
            if kind == "follow":
                return f"you can only follow a person, not {target!r}"
            t.goal = dest
            t.label = f"walking to {target}"
        return t
    if kind == "explore":
        d = DIRECTIONS.get(target.lower().replace(" ", "").replace("-", ""))
        if not d:
            return f"explore needs a direction (north, south, east, west, ...), not {target!r}"
        n = math.hypot(*d)
        gx = min(sim.world.size - 1, max(0, round(a.x + d[0] / n * 18)))
        gy = min(sim.world.size - 1, max(0, round(a.y + d[1] / n * 18)))
        goal = sim.world.nearest_passable(gx, gy, 12)
        if not goal or goal == a.pos:
            return f"there's nowhere further to go {target}"
        t.goal = goal
        t.label = f"exploring {target.lower()}"
        return t
    if kind == "gather":
        if item not in ITEMS:
            # Small models sometimes leave item empty but say it elsewhere ("need water").
            blob = f"{target} {action.get('_hint', '')}".lower()
            item = next((it for it in ITEMS if it[:4] in blob or (it == "berries" and "berr" in blob)), item)
        if item not in ITEMS:
            return f"you can gather wood, stone, berries, fish, crops or water, not {item!r}"
        t.item = item
        cap = WATER_CAPACITY - a.inventory["water"] if item == "water" else a.space()
        if cap <= 0 and not (t.for_id and a.inventory[item]):
            return "you can't carry any more"
        t.amount = max(1, min(amount or 5, cap))
        t.label = f"gathering {item}" + (f" for {sim.agents[t.for_id].name}" if t.for_id else "")
        if t.for_id and a.inventory[item] >= t.amount:
            t.flag, t.count = True, a.inventory[item]  # already have it: just deliver
        return t
    if kind == "drink":
        t.label = "drinking"
        return t
    if kind == "eat":
        if item not in FOOD:
            item = next((f for f in FOOD if a.inventory[f] > 0), "")
        if not item or a.inventory[item] <= 0:
            return "you have no food to eat"
        t.item = item
        t.label = f"eating {item}"
        return t
    if kind == "water":
        if not any(s.kind == "farm" and s.complete for s in sim.world.structures.values()):
            return "there are no farms to water"
        owner = sim.find_agent(target) if target else sim.agents.get(t.for_id or "")
        t.target_id = owner.id if owner else None
        t.label = f"watering {owner.name + chr(39) + 's' if owner else 'a'} farm"
        return t
    if kind == "build":
        if item not in BUILD_SPECS:
            return f"you can build fire, storage, farm, shelter or wall, not {item!r}"
        if target:
            m = re.search(r"(-?\d+)\s*[, ]\s*(-?\d+)", target)
            if m:
                t.goal = (int(m.group(1)), int(m.group(2)))
        t.label = f"building a {item}" + (f" for {sim.agents[t.for_id].name}" if t.for_id else "")
        return t
    if kind == "plan":
        if item not in BUILD_SPECS:
            return f"you can plan fire, storage, farm, shelter or wall, not {item!r}"
        owner = t.for_id or a.id
        open_sites = sum(1 for s in sim.world.structures.values() if s.owner == owner and not s.complete)
        if open_sites >= MAX_OPEN_SITES:
            return f"there are already {open_sites} unfinished projects - get some built first"
        t.amount = max(1, min(amount or 1, 5, MAX_OPEN_SITES - open_sites))
        m = re.search(r"(-?\d+)\s*[, ]\s*(-?\d+)", target)
        t.goal = (int(m.group(1)), int(m.group(2))) if m else a.pos
        t.label = f"marking out {t.amount} {item} site{'s' if t.amount > 1 else ''}"
        return t
    if kind in ORGANIZE:
        t.label = {"found": f"founding {target or 'a group'}", "join": f"joining {target or 'a group'}",
                   "leave": "leaving the group" if not target else f"throwing {target} out", "quit": "quitting your job"}[kind]
        return t
    if kind in ("give", "attack"):
        if kind == "attack" and target.lower().startswith("monster"):
            t.target_id = "monster"
            t.label = "fighting a monster"
            return t
        other = sim.find_agent(target, exclude=a.id)
        if not other:
            return f"there's nobody called {target!r}"
        if not other.alive:
            return f"{other.name} is dead"
        t.target_id = other.id
        if kind == "give":
            if item not in ITEMS:
                return f"you can give wood, stone, berries, fish, crops or water, not {item or 'nothing'!r}"
            cap = WATER_CAPACITY if item == "water" else CAPACITY
            t.amount = max(1, min(amount or max(1, a.inventory[item]), cap))
            # Not carrying enough? Go and get it first (see _give).
            t.label = f"bringing {t.amount} {item} to {other.name}"
        else:
            t.label = f"attacking {other.name}"
        return t
    if kind in ("store", "take"):
        if item and item not in ITEMS:
            return f"there's no such item {item!r}"
        if kind == "store" and item and a.inventory[item] <= 0:
            return f"you don't have any {item}"
        if kind == "store" and not a.load():
            return "you're not carrying anything"
        t.amount = amount
        owner = sim.find_agent(target) if target else None
        if not owner and kind == "store" and t.for_id:
            owner = sim.agents[t.for_id]
        t.target_id = owner.id if owner else None
        t.label = f"{'storing' if kind == 'store' else 'taking'} {item or 'things'}"
        return t
    if kind == "sleep":
        if a.fatigue < 0.2:
            return "you're not tired enough to sleep"
        t.label = "sleeping"
        return t
    if kind == "flee":
        t.label = "running from danger"
        return t
    if kind == "claim":
        t.label = "claiming a structure"
        return t
    return f"unknown action {kind!r}"


def _site_near(sim: Simulation, a: Agent, kind: str, around=None):
    """An unfinished project of this kind near you (or near where you want to build)."""
    here = around or a.pos
    sites = [s for s in sim.world.structures.values() if s.kind == kind and not s.complete and cheb((s.x, s.y), here) <= 8]
    return min(sites, key=lambda s: cheb((s.x, s.y), here), default=None)


def builders(sim: Simulation, site_id: int, exclude: str = "") -> int:
    """How many people are working on this site right now."""
    n = 0
    for o in sim.alive():
        for t in (o.task, o.suspended):
            if o.id != exclude and t and t.kind == "build" and t.status == "active" and t.target_id == site_id:
                n += 1
                break
    return n


def _pick_site(sim: Simulation, a: Agent, t: Task):
    """Which project to work on. For a boss: any of theirs, spreading helpers out rather than piling on one."""
    if t.goal is not None:
        return _site_near(sim, a, t.item, t.goal)
    if t.for_id:
        sites = [s for s in sim.world.structures.values()
                 if s.owner == t.for_id and s.kind == t.item and not s.complete and cheb((s.x, s.y), a.pos) <= 40]
    else:
        sites = [s for s in sim.world.structures.values() if s.kind == t.item and not s.complete and cheb((s.x, s.y), a.pos) <= 8]
    return min(sites, key=lambda s: builders(sim, s.id, a.id) * 8 + cheb((s.x, s.y), a.pos), default=None)


def _next_site(sim: Simulation, a: Agent, t: Task, done):
    """After finishing one of a series (planned sites, or a boss's projects), move on to the next."""
    owner = t.for_id or a.id
    sites = [s for s in sim.world.structures.values()
             if s.owner == owner and not s.complete and s.kind != "remains" and s.id != done.id
             and cheb((s.x, s.y), a.pos) <= 25 and (t.for_id or s.kind == done.kind)]
    return min(sites, key=lambda s: (s.kind != done.kind, builders(sim, s.id, a.id) * 8 + cheb((s.x, s.y), a.pos)), default=None)


# --- execution -----------------------------------------------------------


def walk(sim: Simulation, a: Agent, t: Task, goal: tuple[int, int], near: int = 0, repath_every: int = 0) -> str:
    """Advance one tick toward goal. Returns arrived | moving | blocked."""
    if cheb(a.pos, goal) <= near and (near or a.pos == goal):
        t.path = []
        return "arrived"
    w = sim.world
    if not t.path or (repath_every and (sim.tick - t.started) % repath_every == 0) or not w.passable(*t.path[0]):
        path = w.find_path(a.pos, goal, near=near)
        if path is None:
            return "blocked"
        t.path = path
        if not path:
            return "arrived"
    nxt = t.path[0]
    cost = w.step_cost(a.pos, nxt)
    a.move_points = min(a.move_points + sim.speed(a), max(2.0, cost))
    if a.move_points >= cost:
        a.move_points -= cost
        a.x, a.y = nxt
        t.path.pop(0)
    if cheb(a.pos, goal) <= near and (near or a.pos == goal):
        return "arrived"
    return "moving"


TIME_LIMIT = {"build": 900, "water": 400}


def step(sim: Simulation, a: Agent, t: Task) -> None:
    if t.sub is not None:
        # A delegated step (e.g. fetching wood for a build) runs first.
        step(sim, a, t.sub)
        if t.sub.status == "failed":
            t.fail(t.sub.result)
        elif t.sub.status == "done":
            t.sub = None
            t.path = []
    else:
        fn = _STEPS.get(t.kind)
        if fn:
            fn(sim, a, t)
    if t.status == "active" and sim.tick - t.started > TIME_LIMIT.get(t.kind, 400) and t.kind not in ("follow", "sleep"):
        t.fail("took too long, you gave up")


def _idle(sim, a, t):
    t.timer -= 1
    if t.timer <= 0:
        t.finish("you waited")


def _move(sim, a, t):
    if t.target_id:
        other = sim.agents[t.target_id]
        if not other.alive:
            return t.fail(f"{other.name} is dead")
        r = walk(sim, a, t, other.pos, near=1 if t.kind == "move_to" else 2, repath_every=4)
        if r == "arrived" and t.kind == "move_to":
            t.finish(f"you reached {other.name}")
        elif r == "blocked":
            t.fail(f"you can't find a way to {other.name}")
        elif t.kind == "follow" and sim.tick - t.started > 150:
            t.finish(f"you followed {other.name} for a while")
        return
    r = walk(sim, a, t, t.goal)
    if r == "arrived" and t.kind == "explore":
        t.finish(f"you explored {t.target.lower()} and reached ({a.x}, {a.y})")
    elif r == "arrived":
        t.finish(f"you arrived at {t.target or 'your destination'}")
    elif r == "blocked":
        t.fail(f"there is no way to reach {t.target}")


def _source(sim: Simulation, item: str, p) -> bool:
    w = sim.world
    x, y = p
    if not w.in_bounds(x, y):
        return False
    ch = w.grid[y][x]
    if item == "wood":
        return w.wood.get(p, 0) > 0
    if item == "stone":
        return w.stone.get(p, 0) > 0
    if item == "water":
        return ch in "wo"
    if item == "fish":
        return ch == "-"
    if item == "berries":
        bid = w.bush_at.get(p)
        return bid is not None and w.bushes[bid].berries > 0
    if item == "crops":
        s = w.structure(x, y)
        return bool(s and s.kind == "farm" and s.complete and s.stock > 0)
    return False


def _find_source(sim: Simulation, a: Agent, item: str):
    def match(stand):
        for dx, dy in ((0, 0),) + DIRS8:
            p = (stand[0] + dx, stand[1] + dy)
            if _source(sim, item, p):
                return p
        return None

    return sim.world.search(a.pos, match, max_dist=45)


def _harvest(sim: Simulation, a: Agent, item: str, p) -> int:
    w = sim.world
    if item == "wood":
        return int(w.take_wood(p))
    if item == "stone":
        return int(w.take_stone(p))
    if item == "water":
        return 1
    if item == "fish":
        near_lagoon = any(lm.name == "Fishing Lagoon" and cheb(p, (lm.x, lm.y)) <= 8 for lm in w.landmarks.values())
        return int(sim.rng.random() < (0.8 if near_lagoon else 0.5))
    if item == "berries":
        bid = w.bush_at.get(p)
        if bid is not None and w.bushes[bid].berries > 0:
            w.bushes[bid].berries -= 1
            w.bush_changes.add(bid)
            return 1
    if item == "crops":
        s = w.structure(*p)
        if s and s.kind == "farm" and s.stock > 0:
            s.stock -= 1
            w.structures_dirty = True
            if s.owner and s.owner != a.id and sim.agents[s.owner].alive:
                sim.theft(a, s, "crops", 1)
            return 1
    return 0


def _gather(sim, a, t):
    item = t.item
    if t.flag:
        return _deliver(sim, a, t)
    if t.focus is None or not _source(sim, item, t.focus):
        found = _find_source(sim, a, item)
        if not found:
            if t.count and t.for_id:
                t.flag, t.path = True, []
                return
            if t.count:
                return t.finish(f"you gathered {t.count} {item}; there's no more nearby")
            return t.fail(f"you couldn't find any {item} nearby")
        t.goal, t.focus = found
        t.path = []
        t.progress = 0
    if cheb(a.pos, t.focus) > 1:
        r = walk(sim, a, t, t.goal)
        if r == "blocked":
            t.focus = None
        return
    t.progress += 1
    if t.progress < GATHER_TICKS[item]:
        return
    t.progress = 0
    got = _harvest(sim, a, item, t.focus)
    a.fatigue = min(1.0, a.fatigue + 0.003)
    if got:
        a.inventory[item] += got
        t.count += got
    full = a.inventory["water"] >= WATER_CAPACITY if item == "water" else a.load() >= CAPACITY
    if (t.count >= t.amount or full) and t.for_id:
        t.flag, t.path, t.focus = True, [], None  # now carry it to whoever it's for
    elif t.count >= t.amount or full:
        t.finish(f"you gathered {t.count} {item}")


def _deliver(sim, a, t):
    """Bring what was gathered for someone to their storage, or to them."""
    w = sim.world
    boss = sim.agents[t.for_id]
    if not boss.alive:
        return t.finish(f"you gathered {t.count} {t.item}, but {boss.name} is dead - you keep it")
    have = a.inventory[t.item]
    if have <= 0:
        return t.finish(f"you have no {t.item} left to bring {boss.name}")
    store = w.structures.get(t.focus) if t.focus else None
    if store is None and t.item != "water":
        stores = [s for s in w.structures.values() if s.kind == "storage" and s.complete and s.owner == boss.id
                  and sum(s.items.values()) < STORAGE_CAPACITY and cheb((s.x, s.y), a.pos) <= 40]
        store = min(stores, key=lambda s: cheb((s.x, s.y), a.pos), default=None)
        t.focus = store.id if store else None
    if store:
        r = walk(sim, a, t, (store.x, store.y), near=1)
        if r == "blocked":
            return t.fail(f"you can't reach {boss.name}'s storage")
        if r != "arrived":
            return
        q = min(have, STORAGE_CAPACITY - sum(store.items.values()))
        if q <= 0:
            t.focus = None
            return
        a.inventory[t.item] -= q
        store.items[t.item] += q
        w.structures_dirty = True
        sim.emit(Event(sim.tick, "gift", a.x, a.y, actor=a.id, target=boss.id, data={"item": t.item, "qty": q},
                       text=f"{a.name} put {q} {t.item} in {boss.name}'s storage", radius=6))
        return t.finish(f"you put {q} {t.item} in {boss.name}'s storage")
    r = walk(sim, a, t, boss.pos, near=1, repath_every=3)
    if r == "blocked":
        return t.fail(f"you can't reach {boss.name}")
    if r == "arrived":
        _hand(sim, a, boss, t, have)


def _drink(sim, a, t):
    w = sim.world
    if w.drinkable_near(a.pos):
        a.thirst = max(0.0, a.thirst - 0.25)
        if a.thirst <= 0.05:
            t.finish("you drank your fill")
        return
    if a.inventory["water"] > 0 and not t.flag:
        a.inventory["water"] -= 1
        a.thirst = max(0.0, a.thirst - 0.45)
        if a.thirst <= 0.1 or a.inventory["water"] == 0:
            t.finish("you drank the water you carried")
        return
    t.flag = True  # out of carried water; walk to a source
    if t.goal is None:
        found = _find_source(sim, a, "water")
        if not found:
            known = sim.known_water(a)
            if not known:
                return t.fail("there's no fresh water nearby that you know of")
            t.goal = known[0][1]
        else:
            t.goal = found[0]
    r = walk(sim, a, t, t.goal, near=1 if not w.passable(*t.goal) else 0)
    if r == "blocked":
        t.fail("you can't reach the water")
    elif r == "arrived" and not w.drinkable_near(a.pos):
        t.goal = None


def _eat(sim, a, t):
    if a.inventory[t.item] <= 0:
        other = next((f for f in FOOD if a.inventory[f] > 0), None)
        if not other:
            return t.finish(f"you ate {t.count} and ran out of food")
        t.item = other
    a.inventory[t.item] -= 1
    a.hunger = max(0.0, a.hunger - FOOD[t.item])
    t.count += 1
    if a.hunger <= 0.1:
        t.finish(f"you ate until you were full ({t.count} {t.item})")


def _build(sim, a, t):
    """Start or join a construction project: bring materials, then put in work.

    Anyone can contribute to anyone's project, so "let's build a shelter together" works.
    """
    w = sim.world
    spec = BUILD_SPECS[t.item]
    site = w.structures.get(t.target_id) if isinstance(t.target_id, int) else None
    if site is None:
        site = _pick_site(sim, a, t)
        if site is None:
            if t.repeat:
                return t.fail(f"all of {sim.agents[t.for_id].name}'s {t.item} projects are finished")
            if t.item == "fire" and a.inventory["wood"] >= 2:
                old = next((s for s in w.structures.values() if s.kind == "fire" and s.complete and cheb((s.x, s.y), a.pos) <= 2), None)
                if old:
                    a.inventory["wood"] -= 2
                    old.fuel += spec["fuel"]
                    w.structures_dirty = True
                    return t.finish("you added wood to the fire")
            if t.goal is None:
                t.goal = a.pos
            if cheb(a.pos, t.goal) > 1:
                if walk(sim, a, t, t.goal, near=1) == "blocked":
                    t.fail("you can't get there to build")
                return
            spot = t.goal if w.buildable(*t.goal) and not (t.item == "wall" and t.goal == a.pos) else None
            if spot is None:
                found = w.search(t.goal, lambda p: p if w.buildable(*p) and p != a.pos else None, max_dist=6)
                if not found:
                    return t.fail("there's no room to build here")
                spot = found[1]
            site = w.add_structure(t.item, spot[0], spot[1], t.for_id or a.id, spec["work"])
            site.needs.update(spec["cost"])
            whose = f" for {sim.agents[t.for_id].name}" if t.for_id else ""
            sim.emit(Event(sim.tick, "site", site.x, site.y, actor=a.id,
                           text=f"{a.name} started building a {site.kind}{whose} at ({site.x}, {site.y}). It needs {site.needs_text()}",
                           radius=9))
        t.target_id = site.id
    if site.id not in w.structures:
        return t.fail(f"the {t.item} is gone")
    if cheb(a.pos, (site.x, site.y)) > 1:
        if walk(sim, a, t, (site.x, site.y), near=1) == "blocked":
            t.fail(f"you can't reach the {t.item}")
        return
    if site.complete:
        return t.finish(f"the {site.kind} at ({site.x}, {site.y}) is already finished")
    if site.owner and site.owner != a.id and not t.flag:
        t.flag = True
        sim.emit(Event(sim.tick, "helped_build", a.x, a.y, actor=a.id, target=site.owner,
                       text=f"{a.name} helped {sim.agents[site.owner].name} build a {site.kind}", radius=7))
    if any(v > 0 for v in site.needs.values()):
        given = []
        for k, v in site.needs.items():
            q = min(v, a.inventory[k])
            if q > 0:
                a.inventory[k] -= q
                site.needs[k] -= q
                given.append(f"{q} {k}")
        w.structures_dirty = True
        if given:
            a.log(sim.tick, "done", f"added {', '.join(given)} to the {site.kind} at ({site.x}, {site.y})")
        if site.needs_text():
            # Go and get what's missing, then come back (step() resumes this task afterwards).
            item, qty = max(((k, v) for k, v in site.needs.items() if v > 0), key=lambda kv: kv[1])
            room = a.space()
            if room <= 0:
                return t.fail(f"your hands are full - put something down before fetching {item} for the {site.kind}")
            t.sub = Task("gather", item=item, amount=min(qty, room), label=f"gathering {item} for the {site.kind}", started=sim.tick)
            return
    site.work_done += 1
    a.fatigue = min(1.0, a.fatigue + 0.004)
    w.structures_dirty = True
    if site.complete:
        if site.kind == "fire":
            site.fuel = spec["fuel"]
        owner = sim.agents.get(site.owner)
        sim.emit(Event(sim.tick, "built", site.x, site.y, actor=site.owner or a.id,
                       text=f"{owner.name if owner else a.name} finished a {site.kind} at ({site.x}, {site.y})", radius=9))
        nxt = _next_site(sim, a, t, site)
        if nxt:
            a.log(sim.tick, "done", f"the {site.kind} at ({site.x}, {site.y}) is finished - on to the {nxt.kind} at ({nxt.x}, {nxt.y})")
            t.target_id, t.item, t.goal, t.flag, t.path, t.started = nxt.id, nxt.kind, None, False, [], sim.tick
            return
        t.finish(f"the {site.kind} at ({site.x}, {site.y}) is finished")


def _water(sim, a, t):
    """Fetch water if you carry none, then carry it to a farm and pour it in."""
    w = sim.world
    if a.inventory["water"] == 0 and not t.flag:
        if t.focus is None:
            found = _find_source(sim, a, "water")
            if not found:
                known = sim.known_water(a)
                if not known:
                    return t.fail("you don't know where to get water")
                t.goal = known[0][1]
            else:
                t.goal = found[0]
            t.focus = t.goal
        if w.drinkable_near(a.pos):
            a.inventory["water"] = min(3, a.inventory["water"] + 3)
            t.flag = True
            t.path = []
            return
        if walk(sim, a, t, t.goal, near=0 if w.passable(*t.goal) else 1) == "blocked":
            t.fail("you can't reach the water")
        return
    t.flag = True
    farms = [s for s in w.structures.values() if s.kind == "farm" and s.complete and (not t.target_id or s.owner == t.target_id)]
    own = [s for s in farms if s.owner == a.id]
    farm = min(own or farms, key=lambda s: (s.water, cheb((s.x, s.y), a.pos)), default=None)
    if not farm:
        return t.fail("there's no farm to water")
    if cheb(a.pos, (farm.x, farm.y)) > 1:
        if walk(sim, a, t, (farm.x, farm.y), near=1) == "blocked":
            t.fail("you can't reach the farm")
        return
    poured = 0
    while a.inventory["water"] > 0 and farm.water < 1.0:
        a.inventory["water"] -= 1
        farm.water = min(1.0, farm.water + WATER_PER_UNIT)
        poured += 1
    w.structures_dirty = True
    whose = "your" if farm.owner == a.id else f"{sim.agents[farm.owner].name}'s" if farm.owner else "the"
    t.finish(f"you watered {whose} farm at ({farm.x}, {farm.y})" if poured else f"{whose} farm is already well watered")


def _give(sim, a, t):
    other = sim.agents[t.target_id]
    if not other.alive:
        return t.fail(f"{other.name} is dead")
    if not t.flag:
        t.flag = True
        short = t.amount - a.inventory[t.item]
        room = WATER_CAPACITY - a.inventory["water"] if t.item == "water" else a.space()
        if short > 0 and room > 0:
            # Don't have it (or enough of it): fetch it first, then carry it over.
            t.sub = Task("gather", item=t.item, amount=min(short, room), started=sim.tick,
                         label=f"getting {t.item} for {other.name}")
            return
    r = walk(sim, a, t, other.pos, near=1, repath_every=3)
    if r == "blocked":
        return t.fail(f"you can't reach {other.name}")
    if r == "arrived":
        _hand(sim, a, other, t, t.amount)


def _hand(sim, a, other, t, amount):
    room = WATER_CAPACITY - other.inventory["water"] if t.item == "water" else other.space()
    qty = min(amount, a.inventory[t.item], room)
    if qty <= 0:
        full = f"{other.name} can't carry any more" + (" - they need a storage with room" if t.kind == "gather" else "")
        return t.fail(full if a.inventory[t.item] else f"you have no {t.item} left")
    a.inventory[t.item] -= qty
    other.inventory[t.item] += qty
    sim.emit(Event(sim.tick, "gift", a.x, a.y, actor=a.id, target=other.id, data={"item": t.item, "qty": qty},
                   text=f"{a.name} gave {qty} {t.item} to {other.name}", radius=6))
    t.finish(f"you gave {qty} {t.item} to {other.name}")


def _plan(sim, a, t):
    """Mark out building sites without working on them, so others can build them."""
    w = sim.world
    spec = BUILD_SPECS[t.item]
    owner = t.for_id or a.id
    if cheb(a.pos, t.goal) > sim.vision_radius(a):
        r = walk(sim, a, t, t.goal, near=3)
        if r == "blocked":
            t.fail("you can't get there")
        if r != "arrived":
            return
    gap = 1 if t.item == "wall" else 2  # walls join up; everything else gets elbow room

    def free(p):
        if not w.buildable(*p) or p == a.pos:
            return None
        if gap > 1 and any((p[0] + dx, p[1] + dy) in w.structure_at for dx, dy in DIRS8):
            return None
        return p

    placed = []
    for _ in range(t.amount):
        found = w.search(t.goal, free, max_dist=10)
        if not found:
            break
        x, y = found[1]
        site = w.add_structure(t.item, x, y, owner, spec["work"])
        site.needs.update(spec["cost"])
        placed.append(f"({x}, {y})")
    if not placed:
        return t.fail("there's no room to build there")
    whose = f" for {sim.agents[t.for_id].name}" if t.for_id else ""
    what = f"{len(placed)} {t.item} site{'s' if len(placed) > 1 else ''}"
    sim.emit(Event(sim.tick, "site", a.x, a.y, actor=a.id, radius=12,
                   text=f"{a.name} marked out {what}{whose} at {', '.join(placed)}. Each needs {_cost_text(spec)}"))
    t.finish(f"you marked out {what}{whose} at {', '.join(placed)} - each needs {_cost_text(spec)} and work. Ask people to build them")


def _cost_text(spec) -> str:
    return " and ".join(f"{v} {k}" for k, v in spec["cost"].items())


def _organize(sim, a, t):
    ok, result = sim.organize(a, t.kind, t.target)
    (t.finish if ok else t.fail)(result)


def _storage_for(sim, a, t):
    w = sim.world
    kinds = ("storage",) if t.kind == "store" else ("storage", "remains")

    def ok(s):
        if s.kind not in kinds or not s.complete:
            return False
        if t.target_id and s.owner != t.target_id:
            return False
        return True

    cands = [s for s in w.structures.values() if ok(s) and cheb((s.x, s.y), a.pos) <= 40]
    if t.kind == "store":
        cands = [s for s in cands if sum(s.items.values()) < STORAGE_CAPACITY]
        own = [s for s in cands if s.owner == a.id]
        cands = own or cands
    if t.kind == "take" and t.item:
        cands = [s for s in cands if s.items[t.item] > 0] or cands
    return min(cands, key=lambda s: cheb((s.x, s.y), a.pos), default=None)


def _store_take(sim, a, t):
    w = sim.world
    s = w.structures.get(t.focus) if t.focus else None
    if s is None:
        s = _storage_for(sim, a, t)
        if not s:
            return t.fail("there's no storage with room nearby" if t.kind == "store" else "there's nothing nearby to take from")
        t.focus = s.id
    r = walk(sim, a, t, (s.x, s.y), near=1)
    if r == "blocked":
        return t.fail("you can't reach the storage")
    if r != "arrived":
        return
    items = [t.item] if t.item else list(ITEMS)
    moved = []
    if t.kind == "store":
        for it in items:
            room = STORAGE_CAPACITY - sum(s.items.values())
            q = min(a.inventory[it], t.amount or a.inventory[it], room)
            if q > 0:
                a.inventory[it] -= q
                s.items[it] += q
                moved.append(f"{q} {it}")
    else:
        for it in items:
            room = WATER_CAPACITY - a.inventory["water"] if it == "water" else a.space()
            q = min(s.items[it], t.amount or s.items[it], room)
            if q > 0:
                s.items[it] -= q
                a.inventory[it] += q
                moved.append(f"{q} {it}")
                if s.owner and s.owner != a.id and s.owner in sim.agents and sim.agents[s.owner].alive:
                    sim.theft(a, s, it, q)
    w.structures_dirty = True
    if not moved:
        if t.kind == "store":
            return t.fail("the storage is full" if sum(s.items.values()) >= STORAGE_CAPACITY else f"you have no {t.item or 'things'} to store")
        return t.fail(f"there's no {t.item or 'anything'} left in it")
    owner = sim.agents.get(s.owner)
    whose = "your" if s.owner == a.id else (f"{owner.name}'s" if owner else "the")
    t.finish(f"you {'put' if t.kind == 'store' else 'took'} {', '.join(moved)} {'into' if t.kind == 'store' else 'from'} {whose} {s.kind}")


def _attack(sim, a, t):
    if t.target_id == "monster":
        m = sim.nearest_monster(a, 12)
        if not m:
            return t.finish("there are no monsters near you")
        target_pos, hit = m.pos, lambda: sim.hit_monster(a, m)
    else:
        other = sim.agents[t.target_id]
        if not other.alive:
            return t.finish(f"{other.name} is dead")
        if cheb(a.pos, other.pos) > 15:
            return t.fail(f"{other.name} got away")
        target_pos, hit = other.pos, lambda: sim.hit_agent(a, other)
    if cheb(a.pos, target_pos) > 1:
        t.focus = None
        if walk(sim, a, t, target_pos, near=1, repath_every=2) == "blocked":
            t.fail("you can't reach them")
        return
    t.timer += 1
    if t.timer % 2 == 1:
        hit()
    if t.timer > 40:
        t.finish("you stopped fighting")


def _sleep(sim, a, t):
    a.sleeping = True
    if a.fatigue <= 0.03:
        a.sleeping = False
        t.finish("you woke up rested")


def _flee(sim, a, t):
    w = sim.world
    if t.goal is None:
        threats = [m.pos for m in sim.monsters.values() if cheb(m.pos, a.pos) <= 12]
        threats += [sim.agents[c["source_id"]].pos for c in sim.recent_attackers(a)]
        fire = None
        if sim.is_night():
            fires = [s for s in w.structures.values() if s.burning and cheb((s.x, s.y), a.pos) <= 25]
            fire = min(fires, key=lambda s: cheb((s.x, s.y), a.pos), default=None)
        if fire:
            t.goal = (fire.x, fire.y)
        elif threats:
            tx = sum(p[0] for p in threats) / len(threats)
            ty = sum(p[1] for p in threats) / len(threats)
            dx, dy = a.x - tx, a.y - ty
            n = math.hypot(dx, dy) or 1.0
            t.goal = w.nearest_passable(round(a.x + dx / n * 12), round(a.y + dy / n * 12), 8)
        if not t.goal:
            return t.finish("there's nothing to run from")
    r = walk(sim, a, t, t.goal, near=1)
    if r != "moving":
        t.finish("you got away" if r == "arrived" else "you're cornered")


def _claim(sim, a, t):
    w = sim.world
    s = min((s for s in w.structures.values() if s.complete and s.kind != "remains" and cheb((s.x, s.y), a.pos) <= 3),
            key=lambda s: cheb((s.x, s.y), a.pos), default=None)
    if not s:
        return t.fail("there's nothing here to claim")
    if s.owner == a.id:
        return t.finish(f"the {s.kind} is already yours")
    prev = sim.agents.get(s.owner) if s.owner else None
    s.owner = a.id
    w.structures_dirty = True
    if prev and prev.alive:
        sim.emit(Event(sim.tick, "seize", s.x, s.y, actor=a.id, target=prev.id, data={"kind": s.kind},
                       text=f"{a.name} took {prev.name}'s {s.kind} for themself", radius=9))
        return t.finish(f"you took {prev.name}'s {s.kind}")
    sim.emit(Event(sim.tick, "claim", s.x, s.y, actor=a.id, text=f"{a.name} claimed a {s.kind}", radius=7))
    t.finish(f"the {s.kind} is yours now")


_STEPS = {
    "idle": _idle, "move_to": _move, "follow": _move, "explore": _move, "gather": _gather,
    "drink": _drink, "eat": _eat, "build": _build, "give": _give, "store": _store_take,
    "take": _store_take, "attack": _attack, "sleep": _sleep, "flee": _flee, "claim": _claim,
    "water": _water, "plan": _plan, "found": _organize, "join": _organize, "leave": _organize, "quit": _organize,
}
