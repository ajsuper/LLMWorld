"""Procedural island with a deliberate layout.

The island is built to push survivors into each other:

* A mountain ridge splits it west/east. The only easy crossing is **The Pass**,
  a hilly plateau in the middle that also holds the **Central Spring** - the one
  water source everyone knows about, and high ground that sees far.
* The west is forest: plenty of wood, few berries, and a **Hidden Pond** buried in
  dense forest that you only find by stumbling onto it.
* The east is **Berry Meadows**: lots of food, almost no wood, no water.
* The south coast (where everyone washes up) has fish but no fresh water.
* **South Point** is a peninsula joined by a narrow causeway, **The Neck** - a
  natural fortress for whoever claims it.
* **The Caves** sit at the ridge's northern tip. Monsters come out of them at night,
  which makes the north (and the Hidden Pond) dangerous.
* **Eagle Rock** is a hill on the east coast with a long view.
"""

from __future__ import annotations

import math
import random
from collections import deque

from .noise import fbm
from .world import DIRS8, TERRAINS, Bush, Landmark, World


def generate(size: int = 96, seed: int = 7) -> World:
    N = size
    S = N / 96.0
    rng = random.Random(seed)
    g = [["~"] * N for _ in range(N)]

    def inb(x: int, y: int) -> bool:
        return 2 <= x < N - 2 and 2 <= y < N - 2

    def disc(cx: float, cy: float, r: float, ch: str, only: str | None = None) -> None:
        for y in range(int(cy - r) - 1, int(cy + r) + 2):
            for x in range(int(cx - r) - 1, int(cx + r) + 2):
                if inb(x, y) and math.hypot(x - cx, y - cy) <= r and (only is None or g[y][x] in only):
                    g[y][x] = ch

    def line(x0: float, y0: float, x1: float, y1: float, r: float, ch: str) -> None:
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for i in range(steps + 1):
            t = i / steps
            disc(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r, ch)

    # 1. Main landmass and the South Point peninsula, separated by open water.
    cx, cy = 47 * S, 44 * S
    pen = (79 * S, 86 * S)
    for y in range(N):
        for x in range(N):
            if not inb(x, y):
                continue
            n = fbm(x / (14 * S), y / (14 * S), seed)
            d = math.hypot((x - cx) / (42 * S), (y - cy) / (38 * S))
            pd = math.hypot((x - pen[0]) / (9 * S), (y - pen[1]) / (6 * S))
            if pd < 1.0 + (n - 0.5) * 0.3:
                g[y][x] = ","
            elif d < 0.86 + (n - 0.5) * 0.6 and pd > 1.8:
                g[y][x] = ","

    # 2. The Neck: a thin causeway to South Point.
    neck_a, neck_b = (69 * S, 67 * S), (76 * S, 81 * S)
    line(*neck_a, *neck_b, max(1.0, 1.1 * S), ",")

    # 3. The ridge, running north-south through the middle.
    def rc(y: float) -> float:
        return 47 * S + 3.5 * S * math.sin(y / (8 * S)) + (fbm(0.5, y / (10 * S), seed + 7) - 0.5) * 6 * S

    ytop = min(y for y in range(N) if g[y][int(rc(y))] != "~")
    ybot = int(62 * S)
    for y in range(ytop, ybot + 1):
        c = rc(y)
        taper = min(1.0, (ybot - y) / (10 * S), (y - ytop + 3) / (4 * S))
        hw = (2.2 + 2.5 * fbm(3.3, y / (7 * S), seed + 9)) * S * max(taper, 0.3)
        for x in range(int(c - hw - 5), int(c + hw + 6)):
            if not inb(x, y) or g[y][x] == "~":
                continue
            dx = abs(x - c)
            if dx <= hw:
                g[y][x] = "M"
            elif dx <= hw + 2.2 * S * (0.5 + fbm(x / 5, y / 5, seed + 11)):
                g[y][x] = "r"

    # 4. The Pass: a hilly gap through the ridge, with the spring on top.
    pass_y = int(41 * S)
    for y in range(pass_y - int(6 * S), pass_y + int(6 * S) + 1):
        c = rc(y)
        for x in range(int(c - 12 * S), int(c + 13 * S)):
            if not inb(x, y):
                continue
            e = ((x - c) / (9 * S)) ** 2 + ((y - pass_y) / (3.6 * S)) ** 2
            wobble = (fbm(x / 4, y / 4, seed + 41) - 0.5) * 0.9
            if e < 1.0 + wobble and g[y][x] in "Mr,":
                g[y][x] = "^"
    sx, sy = round(rc(pass_y)), pass_y
    disc(sx + 0.5, sy + 0.5, 3.2 * S, "^", only="Mr,")
    for dx in (0, 1):
        for dy in (0, 1):
            g[sy + dy][sx + dx] = "o"
    spring = (sx, sy)

    # 5. The Caves at the northern tip, with a rocky, dangerous crossing past them.
    cave_y = ytop + int(4 * S)
    cave = (round(rc(cave_y)), cave_y)
    for y in range(cave_y - 1, cave_y + 2):
        c = rc(y)
        for x in range(int(c - 7 * S), int(c + 8 * S)):
            if inb(x, y) and g[y][x] in "M,":
                g[y][x] = "r"
    g[cave[1]][cave[0]] = "C"

    # 6. Biomes: west forest, east meadows, sparse south.
    for y in range(N):
        for x in range(N):
            if g[y][x] != ",":
                continue
            c = rc(y)
            n = fbm(x / (9 * S), y / (9 * S), seed + 21)
            if x < c - 3 * S:
                dens = n + (0.1 if y < 56 * S else -0.06) + max(0.0, (32 * S - x) / (32 * S)) * 0.15
                if dens > 0.71:
                    g[y][x] = "F"
                elif dens > 0.55:
                    g[y][x] = "T"
            elif x > c + 3 * S and 24 * S < y < 70 * S:
                if n > 0.46:
                    g[y][x] = '"'
                elif rng.random() < 0.02:
                    g[y][x] = "T"
            elif n > 0.63:
                g[y][x] = "T"

    # 7. Hidden Pond, wrapped in dense forest.
    pond = (round(19 * S), round(31 * S))
    disc(*pond, 6.5 * S, "F", only=',T"')
    disc(*pond, 1.8 * S, "w")

    # 8. Eagle Rock: a lookout hill on the east coast.
    eagle = (round(79 * S), round(35 * S))
    disc(*eagle, 3.2 * S, "^", only=',"T')
    disc(*eagle, 5.0 * S, "r", only='",')

    # 9. Coast: sand, fishing waters, and the lagoon near the wreck.
    def column_south(x: int) -> int:
        return max(y for y in range(N) if g[y][x] != "~")

    lag_x = round(59 * S)
    lag = (lag_x, column_south(lag_x))
    disc(lag[0], lag[1] - 1, 3.2 * S, "-", only=',."T')

    land_dist = _distance_field(g, N, lambda ch: ch in "~-")
    for y in range(N):
        for x in range(N):
            if g[y][x] in ',"T' and (land_dist[y][x] <= 1 or (land_dist[y][x] == 2 and fbm(x / 4, y / 4, seed + 31) > 0.5)):
                g[y][x] = "."
    sea_dist = _distance_field(g, N, lambda ch: ch not in "~-")
    for y in range(N):
        for x in range(N):
            if g[y][x] == "~" and sea_dist[y][x] <= 2:
                g[y][x] = "-"
    disc(lag[0], lag[1] - 1, 4.5 * S, "-", only="~")

    # 10. The wreck: spawn on the south beach.
    wx = round(44 * S)
    spawn = (wx, column_south(wx))
    while g[spawn[1]][spawn[0]] in "~-":
        spawn = (spawn[0], spawn[1] - 1)

    # 11. Anything unreachable from the wreck becomes rock, so every place can be walked to.
    _seal_unreachable(g, N, spawn)

    # Landmarks. Snapped to walkable tiles where people would stand.
    def snap(p: tuple[int, int], want: str | None = None) -> tuple[int, int]:
        best, bd = p, math.inf
        for y in range(p[1] - 10, p[1] + 11):
            for x in range(p[0] - 10, p[0] + 11):
                if not inb(x, y):
                    continue
                ok = g[y][x] in want if want else TERRAINS[g[y][x]].cost is not None
                if ok and math.hypot(x - p[0], y - p[1]) < bd:
                    best, bd = (x, y), math.hypot(x - p[0], y - p[1])
        return best

    neck_mid = snap((round((neck_a[0] + neck_b[0]) / 2), round((neck_a[1] + neck_b[1]) / 2)))
    landmarks = [
        Landmark("Shipwreck Beach", *spawn, 4),
        Landmark("Central Spring", *spring, 2, kind="water"),
        Landmark("The Pass", *snap((spring[0], spring[1] - 3), "^"), 5),
        Landmark("The Caves", *cave, 3),
        Landmark("Hidden Pond", *pond, 2, kind="water", hidden=True),
        Landmark("Whispering Woods", *snap((round(25 * S), round(48 * S))), 12, kind="area"),
        Landmark("Berry Meadows", *snap((round(66 * S), round(50 * S))), 11, kind="area"),
        Landmark("Eagle Rock", *snap(eagle, "^"), 3),
        Landmark("North Quarry", *snap((round(rc(24 * S) + 6 * S), round(24 * S)), "r"), 4),
        Landmark("Fishing Lagoon", *snap((lag[0], lag[1] - 4)), 4),
        Landmark("The Neck", *neck_mid, 3),
        Landmark("South Point", *snap((round(pen[0]), round(pen[1]))), 6, kind="area"),
    ]

    world = World(N, g, landmarks, spawn, cave)

    # Berry bushes: rich in the meadows, rare elsewhere.
    odds = {'"': 0.07, ",": 0.012, "T": 0.01, ".": 0.004}
    for y in range(N):
        for x in range(N):
            p = odds.get(g[y][x], 0.0)
            if p and rng.random() < p:
                b = Bush(world.new_id(), x, y)
                world.bushes[b.id] = b
                world.bush_at[(x, y)] = b.id
    return world


def _distance_field(g: list[list[str]], N: int, is_source) -> list[list[int]]:
    INF = 10**9
    d = [[INF] * N for _ in range(N)]
    q = deque()
    for y in range(N):
        for x in range(N):
            if is_source(g[y][x]):
                d[y][x] = 0
                q.append((x, y))
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and d[ny][nx] > d[y][x] + 1:
                d[ny][nx] = d[y][x] + 1
                q.append((nx, ny))
    return d


def _seal_unreachable(g: list[list[str]], N: int, start: tuple[int, int]) -> None:
    def walk(x: int, y: int) -> bool:
        return 0 <= x < N and 0 <= y < N and TERRAINS[g[y][x]].cost is not None

    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS8:
            nx, ny = x + dx, y + dy
            if (nx, ny) in seen or not walk(nx, ny):
                continue
            if dx and dy and not (walk(x + dx, y) and walk(x, y + dy)):
                continue
            seen.add((nx, ny))
            q.append((nx, ny))
    for y in range(N):
        for x in range(N):
            if walk(x, y) and (x, y) not in seen:
                wet = any(0 <= x + dx < N and 0 <= y + dy < N and g[y + dy][x + dx] in "~-" for dx, dy in DIRS8)
                g[y][x] = "-" if wet else "M"
