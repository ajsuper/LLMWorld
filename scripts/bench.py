"""Compare models on real LLMWorld prompts.

    uv run python scripts/bench.py qwen3:4b qwen3:8b
    uv run python scripts/bench.py --backend openai --url http://gpu-box:8080 Qwen3-8B

Plays the island forward with scripted brains to get a varied mid-game state, collects
the prompts agents would actually see, then sends the same prompts to each model and
reports validity, sanity of the chosen actions, speed, and how the agents behave.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmworld import prompts  # noqa: E402
from llmworld.brain import Brain, scripted_decision  # noqa: E402
from llmworld.config import ROOT, load_config, load_data  # noqa: E402
from llmworld.sim import Simulation  # noqa: E402
from llmworld.tasks import DIRECTIONS  # noqa: E402
from llmworld.world import BUILD_SPECS  # noqa: E402

ITEM_ACTIONS = {"gather": {"wood", "stone", "berries", "fish", "water"}, "eat": {"berries", "fish"},
                "build": set(BUILD_SPECS), "give": {"wood", "stone", "berries", "fish", "water"}}
TARGET_ACTIONS = {"move_to", "follow", "give", "attack", "explore"}


def collect(cfg: dict, n: int, seed: int) -> tuple[Simulation, list[dict]]:
    sim = Simulation(cfg, load_data())
    rng = random.Random(seed)
    cases = []
    tick = 0
    while len(cases) < n and tick < 3000:
        sim.step()
        tick += 1
        for _, aid, _ in sim.think_requests():
            a = sim.agents[aid]
            system, user = prompts.system_prompt(sim, a), prompts.turn_prompt(sim, a)
            # Past the first stretch, sample ~1 in 6 thoughts so the set spans days, nights and moods.
            if tick > 60 and rng.random() < 0.16 and len(cases) < n:
                names = {o.name for o in sim.agents.values()}
                cases.append({"agent": a.name, "tick": tick, "night": sim.is_night(), "system": system, "user": user,
                              "heard": sorted(sim.agents[s].name for s in a.heard_from), "names": sorted(names),
                              "requests": len(a.pending_requests)})
            sim.pending.append((aid, scripted_decision(sim, a, rng), {}))
    return sim, cases


def check(d: dict, case: dict) -> list[str]:
    """Static sanity checks on a parsed decision. Returns problems found."""
    problems = []
    act = d["action"]
    kind = act.get("type")
    item = str(act.get("item", "")).lower()
    target = str(act.get("target", "")).strip()
    if kind in ITEM_ACTIONS and item not in ITEM_ACTIONS[kind]:
        problems.append(f"{kind} with item {item!r}")
    if kind in TARGET_ACTIONS and not target:
        problems.append(f"{kind} without a target")
    if kind == "explore" and target.lower().replace(" ", "") not in DIRECTIONS:
        problems.append(f"explore toward {target!r}")
    for r in d["reactions"]:
        if r.get("to") not in case["names"]:
            problems.append(f"reaction to unknown {r.get('to')!r}")
        elif r.get("to") not in case["heard"]:
            problems.append(f"reaction to {r.get('to')} who didn't speak")
    if case["requests"] and not d["request_replies"]:
        problems.append("ignored a request")
    say = d.get("say") or {}
    if say.get("to") and say["to"] not in case["names"]:
        problems.append(f"spoke to unknown {say['to']!r}")
    return problems


async def run_model(brain: Brain, model: str, cases: list[dict], parallel: int) -> dict:
    sem = asyncio.Semaphore(parallel)
    results = [None] * len(cases)

    async def one(i: int, c: dict) -> None:
        async with sem:
            t0 = time.monotonic()
            try:
                raw, toks = await brain._complete(c["system"], c["user"], model)
            except Exception as e:  # noqa: BLE001
                results[i] = {"error": f"{type(e).__name__}: {e}"[:200], "latency": time.monotonic() - t0}
                return
            lat = time.monotonic() - t0
            try:
                d = prompts.parse_decision(raw)
            except Exception as e:  # noqa: BLE001
                results[i] = {"raw": raw, "invalid": str(e), "latency": lat, "tokens": toks}
                return
            results[i] = {"raw": raw, "decision": d, "latency": lat, "tokens": toks, "problems": check(d, c)}

    await brain._complete(cases[0]["system"], cases[0]["user"], model)  # load the model before timing
    t0 = time.monotonic()
    await asyncio.gather(*(one(i, c) for i, c in enumerate(cases)))
    return {"model": model, "wall": time.monotonic() - t0, "results": results}


def report(run: dict, cases: list[dict]) -> dict:
    rs = run["results"]
    ok = [r for r in rs if "decision" in r]
    errors = sum(1 for r in rs if "error" in r)
    invalid = sum(1 for r in rs if "invalid" in r)
    clean = sum(1 for r in ok if not r["problems"])
    lat = [r["latency"] for r in rs if "latency" in r]
    toks = sum(r.get("tokens", 0) for r in rs)
    acts = Counter(r["decision"]["action"]["type"] for r in ok)
    problems = Counter(p.split(" ")[0] + " " + " ".join(p.split(" ")[1:3]) for r in ok for p in r["problems"])
    speaks = sum(1 for r in ok if (r["decision"]["say"] or {}).get("text"))
    notes = sum(1 for r in ok if (r["decision"]["note"] or {}).get("op") not in (None, "", "none"))
    asks = sum(1 for r in ok if (r["decision"]["request"] or {}).get("to"))
    felt = Counter(x.get("felt") for r in ok for x in r["decision"]["reactions"])
    n = len(rs)
    return {
        "model": run["model"],
        "valid json": f"{len(ok)}/{n}",
        "no problems": f"{clean}/{n}",
        "errors": errors + invalid,
        "median latency": f"{statistics.median(lat):.1f}s" if lat else "-",
        "p90 latency": f"{sorted(lat)[int(len(lat) * 0.9) - 1]:.1f}s" if lat else "-",
        "throughput": f"{toks / run['wall']:.0f} tok/s, {n / run['wall'] * 60:.0f} thoughts/min",
        "speaks": f"{100 * speaks // max(1, len(ok))}%",
        "requests": f"{100 * asks // max(1, len(ok))}%",
        "notes": f"{100 * notes // max(1, len(ok))}%",
        "actions": dict(acts.most_common(8)),
        "feelings": dict(felt.most_common(6)),
        "problems": dict(problems.most_common(6)),
    }


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("-n", type=int, default=40, help="number of prompts")
    p.add_argument("--parallel", type=int)
    p.add_argument("--backend", choices=["ollama", "openai"])
    p.add_argument("--url")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--samples", type=int, default=3, help="side-by-side examples to print")
    args = p.parse_args()

    cfg = load_config()
    for key, val in (("backend", args.backend), ("url", args.url), ("parallel", args.parallel)):
        if val is not None:
            cfg["llm"][key] = val
    if cfg["llm"]["backend"] == "scripted":
        cfg["llm"]["backend"] = "ollama"
    sim, cases = collect(cfg, args.n, args.seed)
    print(f"Collected {len(cases)} prompts from {sim.tick} ticks of play "
          f"({sum(c['night'] for c in cases)} at night, {sum(bool(c['heard']) for c in cases)} after hearing someone).\n")

    brain = Brain(sim, cfg)
    await brain.start()
    runs = []
    try:
        for m in args.models:
            print(f"Running {m} ({cfg['llm']['parallel']} at a time)...", flush=True)
            runs.append(await run_model(brain, m, cases, cfg["llm"]["parallel"]))
    finally:
        await brain.close()

    reports = [report(r, cases) for r in runs]
    keys = [k for k in reports[0] if k not in ("model", "actions", "feelings", "problems")]
    w = max(len(r["model"]) for r in reports) + 2
    print("\n" + " " * 16 + "".join(r["model"].ljust(max(w, 34)) for r in reports))
    for k in keys:
        print(k.ljust(16) + "".join(str(r[k]).ljust(max(w, 34)) for r in reports))
    for k in ("actions", "feelings", "problems"):
        print(f"\n{k}:")
        for r in reports:
            print(f"  {r['model']}: {r[k] or '-'}")

    rng = random.Random(1)
    picks = rng.sample(range(len(cases)), min(args.samples, len(cases)))
    for i in picks:
        c = cases[i]
        print(f"\n--- {c['agent']}, tick {c['tick']}{', night' if c['night'] else ''} " + "-" * 30)
        heard = [ln for ln in c["user"].splitlines() if ln.startswith("- ") and '"' in ln][:3]
        for ln in heard:
            print(f"  heard {ln[2:]}")
        for run in runs:
            r = run["results"][i]
            if "decision" not in r:
                print(f"  [{run['model']}] {r.get('error') or r.get('invalid')}")
                continue
            d = r["decision"]
            act = d["action"]
            say = (d["say"] or {}).get("text")
            print(f"  [{run['model']}] thinks: {d['thought']}")
            print(f"  {' ' * (len(run['model']) + 2)} does: {act['type']} {act.get('target', '')} {act.get('item', '')}".rstrip())
            if say:
                print(f"  {' ' * (len(run['model']) + 2)} says: \"{say}\"")

    out = ROOT / "runs" / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"reports": reports, "cases": cases, "runs": runs}, indent=1, default=str))
    print(f"\nFull results: {out}")


if __name__ == "__main__":
    asyncio.run(main())
