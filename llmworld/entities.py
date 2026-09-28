"""Agents, monsters, tasks and events."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

EMOTIONS = ("joy", "sadness", "anger", "fear")
ITEMS = ("wood", "stone", "berries", "fish", "crops", "water")
CAPACITY = 20
WATER_CAPACITY = 3
NOTEPAD_LINES = 10
NOTEPAD_WIDTH = 100


@dataclass
class Relation:
    affinity: float = 0.0  # -1 dislike .. 1 like
    trust: float = 0.0  # -1 .. 1
    fear: float = 0.0  # 0 .. 1
    respect: float = 0.0  # 0 .. 1
    met: bool = False
    # Requests *I* made of them, and how they answered.
    asked: int = 0
    obeyed: int = 0
    refused: int = 0
    last_answer: str = ""


@dataclass
class Task:
    kind: str
    target: str = ""
    item: str = ""
    amount: int = 0
    label: str = ""
    goal: tuple[int, int] | None = None
    focus: tuple[int, int] | None = None
    target_id: str | int | None = None
    path: list = field(default_factory=list)
    progress: int = 0
    count: int = 0
    timer: int = 0
    started: int = 0
    status: str = "active"  # active | done | failed
    result: str = ""
    flag: bool = False
    sub: "Task | None" = None  # a step this task delegated (e.g. a build fetching its wood)
    for_id: str | None = None  # doing this for someone: what's built is theirs, what's gathered goes to them
    repeat: bool = False  # part of a standing job: works on what exists, never starts new projects

    def finish(self, result: str) -> None:
        self.status, self.result = "done", result

    def fail(self, reason: str) -> None:
        self.status, self.result = "failed", reason


@dataclass
class Event:
    tick: int
    kind: str
    x: int
    y: int
    actor: str | None = None
    target: str | None = None
    text: str = ""  # third-person description, used for witnesses and the world feed
    data: dict = field(default_factory=dict)
    radius: float = 8.0
    sound: bool = False  # heard rather than seen: no line of sight needed
    public: bool = True


@dataclass
class Agent:
    id: str
    name: str
    color: str
    tier: str
    traits: list[str]
    ambition: str | None
    backstory: str
    x: int
    y: int
    hp: float = 1.0
    alive: bool = True
    death_cause: str = ""
    thirst: float = 0.3
    hunger: float = 0.2
    fatigue: float = 0.1
    emotions: dict = field(default_factory=lambda: {e: 0.0 for e in EMOTIONS})
    causes: list = field(default_factory=list)  # {emotion, source, reason, tick, strength}
    relations: dict = field(default_factory=dict)  # other id -> Relation
    inventory: Counter = field(default_factory=Counter)
    notepad: list = field(default_factory=list)
    known: set = field(default_factory=set)  # landmark names
    task: Task | None = None
    move_points: float = 0.0
    sleeping: bool = False
    history: deque = field(default_factory=lambda: deque(maxlen=400))
    inbox: list = field(default_factory=list)  # {text, salient, kind, speaker}
    recent: deque = field(default_factory=lambda: deque(maxlen=6))  # own recent actions, for the prompt
    said: deque = field(default_factory=lambda: deque(maxlen=5))  # own recent words, for the prompt
    pending_requests: list = field(default_factory=list)  # {from, task, tick}
    heard_from: set = field(default_factory=set)  # speakers included in the last prompt
    thinking: bool = False
    queued: bool = False
    think_reason: str = ""
    last_think: int = -999
    need_flags: set = field(default_factory=set)
    seen_bodies: set = field(default_factory=set)
    last_monster_alert: int = -999
    bubble: tuple | None = None  # (text, volume, until_tick)
    last_prompt: str = ""
    last_raw: str = ""
    think_count: int = 0
    suspended: Task | None = None  # job paused while the body drinks or eats
    promises: list = field(default_factory=list)  # requests this agent accepted: {to, task, tick}
    last_request: int = -999
    memories: list = field(default_factory=list)  # long-term memory, rewritten periodically by the agent
    memory_tick: int = 0
    remembering: bool = False
    group: str | None = None  # name of the group this agent belongs to
    duty: dict | None = None  # standing job: {action, for, task, since}; resumed whenever idle
    duty_wait: int = 0  # tick before which an unworkable duty isn't retried
    purpose: str = ""  # what they want now, in their own words, once they've changed their mind
    purpose_why: str = ""
    purpose_tick: int = -999
    past_purposes: list = field(default_factory=list)  # {t, was, now, why}

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    def rel(self, other_id: str) -> Relation:
        r = self.relations.get(other_id)
        if r is None:
            r = self.relations[other_id] = Relation()
        return r

    def load(self) -> int:
        return sum(self.inventory.values())

    def space(self) -> int:
        return max(0, CAPACITY - self.load())

    def log(self, tick: int, kind: str, text: str) -> None:
        self.history.append({"t": tick, "k": kind, "text": text})


@dataclass
class Group:
    name: str
    leader: str
    members: set = field(default_factory=set)  # includes the leader
    founded: int = 0


@dataclass
class Monster:
    id: int
    x: int
    y: int
    hp: float = 1.0
    target: str | None = None
    path: list = field(default_factory=list)
    move_points: float = 0.0
    cooldown: int = 0
    repath: int = 0

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)
