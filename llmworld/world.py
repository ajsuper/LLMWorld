"""The tile world: terrain, resources, structures, pathfinding and line of sight."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter, deque
from dataclasses import dataclass, field

Pos = tuple[int, int]


@dataclass(frozen=True)
class Terrain:
    char: str
    name: str
    cost: float | None  # None = impassable
    blocks_los: bool = False
    drinkable: bool = False
    elevated: bool = False
    wood: int = 0
    stone: int = 0


TERRAINS: dict[str, Terrain] = {
    t.char: t
    for t in (
        Terrain("~", "the sea", None),
        Terrain("-", "fishing waters", None),
        Terrain(".", "sand", 1.0),
        Terrain(",", "grass", 1.0),
        Terrain('"', "meadow", 1.0),
        Terrain("T", "forest", 1.5, wood=4),
        Terrain("F", "dense forest", 2.2, blocks_los=True, wood=6),
        Terrain("^", "hills", 1.5, elevated=True),
        Terrain("M", "mountain", None, blocks_los=True),
        Terrain("r", "rocky ground", 1.4, stone=5),
        Terrain("w", "pond", None, drinkable=True),
        Terrain("o", "spring", None, drinkable=True),
        Terrain("C", "cave mouth", 1.0),
    )
}

DIRS8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

BUILD_SPECS: dict[str, dict] = {
    "fire": {"cost": {"wood": 3}, "work": 3, "fuel": 240},
    "storage": {"cost": {"wood": 4}, "work": 5},
    "farm": {"cost": {"wood": 2, "berries": 2}, "work": 6},
    "shelter": {"cost": {"wood": 6, "stone": 3}, "work": 10},
    "wall": {"cost": {"stone": 2}, "work": 3},
}
# Farms: a watered farm grows a crop in FARM_GROW_TICKS (faster on meadow), yielding FARM_YIELD
# crops. Water drains over FARM_WATER_TICKS; a farm left dry for FARM_WITHER_TICKS loses its crop.
FARM_GROW_TICKS = 150
FARM_YIELD = 6
FARM_WATER_TICKS = 120
FARM_WITHER_TICKS = 150
WATER_PER_UNIT = 0.5
STORAGE_CAPACITY = 40
FIRE_LIGHT_RADIUS = 4.5


def cheb(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def dist(a: Pos, b: Pos) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Landmark:
    name: str
    x: int
    y: int
    radius: float
    kind: str = "place"  # place | water | area
    hidden: bool = False


@dataclass
class Bush:
    id: int
    x: int
    y: int
    berries: int = 4
    max: int = 4
    regrow: int = 0


@dataclass
class Structure:
    id: int
    kind: str
    x: int
    y: int
    owner: str | None = None
    work_needed: int = 1
    work_done: int = 0
    fuel: int = 0
    stock: int = 0  # ripe crops on a farm
    grow: int = 0
    needs: Counter = field(default_factory=Counter)  # materials still missing before work can start
    water: float = 0.0  # farm soil moisture, 0..1
    growth: float = 0.0  # farm crop progress, 0..1
    dry: int = 0  # ticks a farm has gone without water
    items: Counter = field(default_factory=Counter)
    thefts: list = field(default_factory=list)  # (thief_id, item, qty) the owner hasn't noticed yet
    of: str | None = None  # remains: whose body this is

    @property
    def complete(self) -> bool:
        return self.work_done >= self.work_needed

    def needs_text(self) -> str:
        return ", ".join(f"{v} {k}" for k, v in self.needs.items() if v > 0)

    @property
    def burning(self) -> bool:
        return self.kind == "fire" and self.complete and self.fuel > 0


class World:
    def __init__(self, size: int, grid: list[list[str]], landmarks: list[Landmark], spawn: Pos, cave: Pos):
        self.size = size
        self.grid = grid
        self.landmarks: dict[str, Landmark] = {lm.name: lm for lm in landmarks}
        self.spawn = spawn
        self.cave = cave
        self.wood: dict[Pos, int] = {}
        self.stone: dict[Pos, int] = {}
        self.forest_origin: list[Pos] = []
        for y in range(size):
            for x in range(size):
                t = TERRAINS[grid[y][x]]
                if t.wood:
                    self.wood[(x, y)] = t.wood
                    self.forest_origin.append((x, y))
                if t.stone:
                    self.stone[(x, y)] = t.stone
        self.bushes: dict[int, Bush] = {}
        self.bush_at: dict[Pos, int] = {}
        self.structures: dict[int, Structure] = {}
        self.structure_at: dict[Pos, int] = {}
        self.tile_changes: list[tuple[int, int, str]] = []
        self.bush_changes: set[int] = set()
        self.structures_dirty = True
        self._ids = itertools.count(1)

    # --- basic queries -------------------------------------------------

    def new_id(self) -> int:
        return next(self._ids)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def tile(self, x: int, y: int) -> str:
        return self.grid[y][x]

    def terrain(self, x: int, y: int) -> Terrain:
        return TERRAINS[self.grid[y][x]]

    def set_tile(self, x: int, y: int, ch: str) -> None:
        self.grid[y][x] = ch
        self.tile_changes.append((x, y, ch))

    def structure(self, x: int, y: int) -> Structure | None:
        sid = self.structure_at.get((x, y))
        return self.structures.get(sid) if sid is not None else None

    def passable(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y) or TERRAINS[self.grid[y][x]].cost is None:
            return False
        s = self.structure(x, y)
        return not (s and s.kind == "wall" and s.complete)

    def buildable(self, x: int, y: int) -> bool:
        return self.passable(x, y) and (x, y) not in self.structure_at and self.grid[y][x] not in "C"

    def drinkable_near(self, pos: Pos) -> Pos | None:
        for dx, dy in ((0, 0),) + DIRS8:
            x, y = pos[0] + dx, pos[1] + dy
            if self.in_bounds(x, y) and TERRAINS[self.grid[y][x]].drinkable:
                return (x, y)
        return None

    def lit(self, x: int, y: int) -> bool:
        return any(s.burning and dist((x, y), (s.x, s.y)) <= FIRE_LIGHT_RADIUS for s in self.structures.values())

    def in_shelter(self, x: int, y: int) -> bool:
        s = self.structure(x, y)
        return bool(s and s.kind == "shelter" and s.complete)

    # --- structures ------------------------------------------------------

    def add_structure(self, kind: str, x: int, y: int, owner: str | None, work: int) -> Structure:
        s = Structure(self.new_id(), kind, x, y, owner=owner, work_needed=work)
        self.structures[s.id] = s
        self.structure_at[(x, y)] = s.id
        self.structures_dirty = True
        return s

    def remove_structure(self, s: Structure) -> None:
        self.structures.pop(s.id, None)
        if self.structure_at.get((s.x, s.y)) == s.id:
            del self.structure_at[(s.x, s.y)]
        self.structures_dirty = True

    # --- resources -------------------------------------------------------

    def take_wood(self, pos: Pos) -> bool:
        n = self.wood.get(pos, 0)
        if n <= 0:
            return False
        n -= 1
        if n > 0:
            self.wood[pos] = n
        elif self.grid[pos[1]][pos[0]] == "F":
            self.set_tile(*pos, "T")
            self.wood[pos] = TERRAINS["T"].wood
        else:
            del self.wood[pos]
            self.set_tile(*pos, ",")
        return True

    def take_stone(self, pos: Pos) -> bool:
        n = self.stone.get(pos, 0)
        if n <= 0:
            return False
        if n > 1:
            self.stone[pos] = n - 1
        else:
            del self.stone[pos]
            self.set_tile(*pos, ",")
        return True

    # --- movement --------------------------------------------------------

    def neighbors(self, x: int, y: int):
        for dx, dy in DIRS8:
            nx, ny = x + dx, y + dy
            if not self.passable(nx, ny):
                continue
            if dx and dy and not (self.passable(x + dx, y) and self.passable(x, y + dy)):
                continue
            yield (nx, ny)

    def step_cost(self, a: Pos, b: Pos) -> float:
        base = TERRAINS[self.grid[b[1]][b[0]]].cost or 1.0
        return base * (1.41 if a[0] != b[0] and a[1] != b[1] else 1.0)

    def find_path(self, start: Pos, goal: Pos, near: int = 0, max_expand: int = 9000) -> list[Pos] | None:
        """A* to any tile within Chebyshev distance `near` of goal. Returns steps excluding start."""
        if near == 0 and not self.passable(*goal):
            near = 1
        if cheb(start, goal) <= near:
            return []
        gx, gy = goal

        def h(p: Pos) -> float:
            dx, dy = abs(p[0] - gx), abs(p[1] - gy)
            return max(0, max(dx, dy) - near) + 0.41 * min(dx, dy)

        heap = [(h(start), 0.0, start)]
        came: dict[Pos, Pos | None] = {start: None}
        cost = {start: 0.0}
        expanded = 0
        while heap and expanded < max_expand:
            _, g, cur = heapq.heappop(heap)
            if g > cost[cur]:
                continue
            expanded += 1
            if cheb(cur, goal) <= near:
                path = []
                while cur != start:
                    path.append(cur)
                    cur = came[cur]
                return path[::-1]
            for nb in self.neighbors(*cur):
                ng = g + self.step_cost(cur, nb)
                if ng < cost.get(nb, math.inf):
                    cost[nb] = ng
                    came[nb] = cur
                    heapq.heappush(heap, (ng + h(nb), ng, nb))
        return None

    def search(self, start: Pos, match, max_dist: int = 40):
        """BFS over walkable tiles; match(pos) returns a target or None. Returns (stand, target)."""
        seen = {start}
        q = deque([(start, 0)])
        while q:
            p, d = q.popleft()
            t = match(p)
            if t is not None:
                return p, t
            if d >= max_dist:
                continue
            for nb in self.neighbors(*p):
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return None

    def nearest_passable(self, x: int, y: int, radius: int = 10) -> Pos | None:
        best, best_d = None, math.inf
        for yy in range(y - radius, y + radius + 1):
            for xx in range(x - radius, x + radius + 1):
                if self.passable(xx, yy):
                    d = dist((x, y), (xx, yy))
                    if d < best_d:
                        best, best_d = (xx, yy), d
        return best

    # --- sight -----------------------------------------------------------

    def line_of_sight(self, a: Pos, b: Pos, elevated: bool = False) -> bool:
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy
        trees = 0
        x, y = x0, y0
        while True:
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            if (x, y) == (x1, y1):
                return True
            ch = self.grid[y][x]
            if ch == "M":
                return False
            s = self.structure(x, y)
            if s and s.kind == "wall" and s.complete:
                return False
            if not elevated:
                if ch == "F":
                    return False
                if ch == "T":
                    trees += 1
                    if trees > 2:
                        return False
