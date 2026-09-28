"""Web server: runs the tick loop and streams the world to browsers over a WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from aiohttp import WSMsgType, web

from . import snapshot
from .brain import Brain
from .sim import Simulation

log = logging.getLogger("llmworld.server")
WEB = Path(__file__).resolve().parent.parent / "web"


class Server:
    def __init__(self, sim: Simulation, brain: Brain, cfg: dict):
        self.sim = sim
        self.brain = brain
        self.cfg = cfg
        self.tick_seconds = float(cfg["world"]["tick_seconds"])
        self.paused = False
        self.step_once = False
        self.clients: dict[web.WebSocketResponse, dict] = {}
        self.app = web.Application()
        self.app.router.add_get("/", self._index)
        self.app.router.add_get("/ws", self._ws)
        self.app.router.add_static("/static", WEB)

    async def _index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB / "index.html")

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
        await ws.prepare(request)
        self.clients[ws] = {"inspect": None}
        await ws.send_str(json.dumps(snapshot.init_payload(self.sim)))
        await ws.send_str(json.dumps(snapshot.tick_payload(self.sim, self.brain, self.paused)))
        await ws.send_str(json.dumps(snapshot.social_graph(self.sim)))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                try:
                    await self._handle(ws, data)
                except Exception:
                    log.exception("failed to handle %s", data.get("type"))
        finally:
            self.clients.pop(ws, None)
        return ws

    async def _handle(self, ws: web.WebSocketResponse, data: dict) -> None:
        kind = data.get("type")
        if kind == "inspect":
            aid = data.get("id")
            self.clients[ws]["inspect"] = aid if aid in self.sim.agents else None
            if aid in self.sim.agents:
                await ws.send_str(json.dumps(snapshot.agent_detail(self.sim, self.sim.agents[aid])))
        elif kind == "control":
            action = data.get("action")
            if action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            elif action == "step":
                self.step_once = True
            elif action == "speed":
                self.tick_seconds = max(0.25, min(10.0, float(data.get("value", self.tick_seconds))))
            await self._broadcast_tick()

    async def _broadcast_tick(self) -> None:
        if not self.clients:
            self.sim.world.structures_dirty = False
            return
        payload = snapshot.tick_payload(self.sim, self.brain, self.paused)
        payload["tick_seconds"] = self.tick_seconds
        self.sim.world.structures_dirty = False
        msg = json.dumps(payload)
        social = json.dumps(snapshot.social_graph(self.sim)) if self.sim.tick % 5 == 0 else None
        for ws, state in list(self.clients.items()):
            try:
                await ws.send_str(msg)
                if social:
                    await ws.send_str(social)
                aid = state["inspect"]
                if aid:
                    await ws.send_str(json.dumps(snapshot.agent_detail(self.sim, self.sim.agents[aid])))
            except (ConnectionResetError, RuntimeError):
                self.clients.pop(ws, None)
            except Exception:
                log.exception("failed to send to a client")

    async def tick_loop(self) -> None:
        while True:
            t0 = time.monotonic()
            if not self.paused or self.step_once:
                self.step_once = False
                try:
                    self.sim.step()
                    self.brain.schedule()
                except Exception:
                    log.exception("tick %d failed", self.sim.tick)
                    self.paused = True
                await self._broadcast_tick()
            await asyncio.sleep(max(0.05, self.tick_seconds - (time.monotonic() - t0)))

    async def run(self) -> None:
        await self.brain.start()
        runner = web.AppRunner(self.app)
        await runner.setup()
        sc = self.cfg["server"]
        await web.TCPSite(runner, sc["host"], sc["port"]).start()
        log.info("LLMWorld running at http://%s:%s  (brain: %s)", sc["host"], sc["port"], self.brain.backend)
        try:
            await self.tick_loop()
        finally:
            await self.brain.close()
            await runner.cleanup()
            self.sim.close()
