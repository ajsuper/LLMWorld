"""Run LLMWorld: python -m llmworld [--scripted] [--backend ...] [--url ...]"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .brain import Brain
from .config import ROOT, load_config, load_data
from .server import Server
from .sim import Simulation, new_run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="An island sandbox for LLM agents.")
    p.add_argument("--config", help="path to config.yaml")
    p.add_argument("--scripted", action="store_true", help="no LLM: use simple scripted brains")
    p.add_argument("--backend", choices=["ollama", "openai", "scripted"])
    p.add_argument("--url", help="LLM server base URL")
    p.add_argument("--main-model", help="model for agents with an ambition")
    p.add_argument("--fast-model", help="model for everyone else")
    p.add_argument("--parallel", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--tick", type=float, help="real seconds per simulation tick (raise it on slow machines)")
    p.add_argument("--port", type=int)
    p.add_argument("--host")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    cfg = load_config(args.config and __import__("pathlib").Path(args.config))
    if args.scripted:
        cfg["llm"]["backend"] = "scripted"
    for key, val in (("backend", args.backend), ("url", args.url), ("parallel", args.parallel)):
        if val is not None:
            cfg["llm"][key] = val
    if args.main_model:
        cfg["llm"]["models"]["main"] = args.main_model
    if args.fast_model:
        cfg["llm"]["models"]["fast"] = args.fast_model
    if args.tick:
        cfg["world"]["tick_seconds"] = args.tick
    if args.seed is not None:
        cfg["world"]["seed"] = args.seed
    if args.port:
        cfg["server"]["port"] = args.port
    if args.host:
        cfg["server"]["host"] = args.host

    sim = Simulation(cfg, load_data(), run_dir=new_run_dir(ROOT / "runs"))
    server = Server(sim, Brain(sim, cfg), cfg)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
