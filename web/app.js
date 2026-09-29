"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const TILE = 16; // pixels per tile in the pre-rendered terrain
const TERRAIN = {
  "~": "#143047", "-": "#1f5572", ".": "#d4c28e", ",": "#6c9146", '"': "#8aac4f",
  "T": "#44703a", "F": "#2b4e2c", "^": "#7f8757", "M": "#6a625b", "r": "#8b8277",
  "w": "#3d9ccc", "o": "#6fd3f3", "C": "#1c1614",
};

const S = {
  size: 0, grid: [], meta: {}, agents: {}, monsters: [], structures: [], bushes: new Map(),
  landmarks: [], feed: [], tick: 0, tod: 0, night: false, phase: "", paused: false,
  tickSeconds: 2, tickAt: 0, selected: null, follow: false, detail: null, nightFraction: 0.35,
};
const cam = { x: 0, y: 0, scale: 8 }; // x,y = world tile at the canvas's top-left; scale = px per tile

const $ = (id) => document.getElementById(id);
const canvas = $("map"), ctx = canvas.getContext("2d");
const terrain = document.createElement("canvas"), tctx = terrain.getContext("2d");
const shade = document.createElement("canvas"), sctx = shade.getContext("2d");
let ws = null;

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "init") onInit(m);
    else if (m.type === "tick") onTick(m);
    else if (m.type === "detail" && m.id === S.selected) { S.detail = m; renderMind(); }
    else if (m.type === "social") { S.social = m; renderPower(); }
  };
  ws.onclose = () => { $("clock-text").textContent = "Disconnected. Retrying…"; setTimeout(connect, 1500); };
}
const send = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));

function onInit(m) {
  S.size = m.size;
  S.grid = m.tiles.map((r) => r.split(""));
  S.landmarks = m.landmarks;
  S.tickSeconds = m.tick_seconds;
  S.nightFraction = m.night_fraction;
  S.meta = Object.fromEntries(m.agents.map((a) => [a.id, a]));
  S.bushes = new Map(m.bushes.map(([id, x, y, n]) => [id, { x, y, n }]));
  S.structures = m.structures;
  S.feed = m.feed;
  terrain.width = terrain.height = S.size * TILE;
  for (let y = 0; y < S.size; y++) for (let x = 0; x < S.size; x++) drawTile(x, y);
  buildRoster();
  if (!S.fitted) { fit(); S.fitted = true; }
  renderIsland();
  if (S.selected) send({ type: "inspect", id: S.selected });
}

function onTick(m) {
  const now = performance.now();
  for (const a of m.agents) {
    const prev = S.agents[a.id];
    const cur = prev ? displayPos(prev, now) : { x: a.x, y: a.y };
    S.agents[a.id] = { ...a, fx: cur.x, fy: cur.y };
  }
  S.tickAt = now;
  S.tick = m.tick; S.tod = m.tod; S.night = m.night; S.phase = m.phase; S.paused = m.paused;
  if (m.tick_seconds) S.tickSeconds = m.tick_seconds;
  S.monsters = m.monsters.map(([id, x, y]) => {
    const old = S.monsterPos?.get(id);
    return { id, x, y, fx: old ? old.x : x, fy: old ? old.y : y };
  });
  S.monsterPos = new Map(S.monsters.map((mo) => [mo.id, { x: mo.x, y: mo.y }]));
  for (const [x, y, ch] of m.tiles) { S.grid[y][x] = ch; drawTile(x, y); }
  for (const [id, n] of m.bushes) { const b = S.bushes.get(id); if (b) b.n = n; }
  if (m.structures) S.structures = m.structures;
  if (m.feed.length) {
    for (const f of m.feed) {
      // A request's line is resent as answers arrive: update it in place.
      const i = f.id ? S.feed.findIndex((x) => x.id === f.id) : -1;
      if (i >= 0) S.feed[i] = f; else S.feed.push(f);
    }
    S.feed = S.feed.slice(-300);
    renderIsland();
  }
  updateBar(m);
  updateRoster();
}

// ---------------------------------------------------------------------------
// Terrain (pre-rendered once, tiles patched as the island changes)
// ---------------------------------------------------------------------------
function hash(x, y, s = 0) {
  let h = (x * 374761393 + y * 668265263 + s * 982451653) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}
function shadeHex(hex, f) {
  const n = parseInt(hex.slice(1), 16);
  const c = (v) => Math.max(0, Math.min(255, Math.round(v * f)));
  return `rgb(${c(n >> 16)},${c((n >> 8) & 255)},${c(n & 255)})`;
}
function drawTile(x, y) {
  const ch = S.grid[y][x], px = x * TILE, py = y * TILE;
  const base = TERRAIN[ch] || "#f0f";
  tctx.fillStyle = shadeHex(base, 0.94 + hash(x, y) * 0.12);
  tctx.fillRect(px, py, TILE, TILE);
  const r = (i) => hash(x, y, i);
  if (ch === "T" || ch === "F") {
    const n = ch === "F" ? 3 : 2;
    for (let i = 0; i < n; i++) {
      tctx.fillStyle = shadeHex(base, ch === "F" ? 0.72 : 0.78);
      tctx.beginPath();
      tctx.arc(px + 3 + r(i) * 10, py + 3 + r(i + 7) * 10, ch === "F" ? 4.2 : 3.4, 0, Math.PI * 2);
      tctx.fill();
      tctx.fillStyle = shadeHex(base, 1.25);
      tctx.beginPath();
      tctx.arc(px + 2 + r(i) * 10, py + 2 + r(i + 7) * 10, 1.3, 0, Math.PI * 2);
      tctx.fill();
    }
  } else if (ch === "M") {
    tctx.fillStyle = shadeHex(base, 1.3);
    tctx.beginPath();
    tctx.moveTo(px + 2, py + 13); tctx.lineTo(px + 8, py + 3 + r(1) * 2); tctx.lineTo(px + 14, py + 13);
    tctx.closePath(); tctx.fill();
    tctx.fillStyle = shadeHex(base, 0.8);
    tctx.beginPath();
    tctx.moveTo(px + 8, py + 3 + r(1) * 2); tctx.lineTo(px + 14, py + 13); tctx.lineTo(px + 9, py + 13);
    tctx.closePath(); tctx.fill();
  } else if (ch === "r") {
    for (let i = 0; i < 3; i++) {
      tctx.fillStyle = shadeHex(base, i % 2 ? 0.75 : 1.2);
      tctx.fillRect(px + r(i) * 12, py + r(i + 3) * 12, 3, 2);
    }
  } else if (ch === "^") {
    tctx.strokeStyle = shadeHex(base, 0.8);
    tctx.lineWidth = 1;
    tctx.beginPath();
    tctx.arc(px + 8, py + 12, 6, Math.PI * 1.15, Math.PI * 1.85);
    tctx.stroke();
  } else if (ch === "~" || ch === "-" || ch === "w") {
    if (r(2) < 0.25) {
      tctx.strokeStyle = shadeHex(base, 1.25);
      tctx.beginPath();
      const wx = px + r(3) * 8, wy = py + 4 + r(4) * 8;
      tctx.moveTo(wx, wy); tctx.quadraticCurveTo(wx + 2.5, wy - 2, wx + 5, wy); tctx.stroke();
    }
  } else if (ch === "o") {
    tctx.fillStyle = "#c8f3ff";
    tctx.fillRect(px + 5 + r(1) * 5, py + 5 + r(2) * 5, 2, 2);
  } else if (ch === "C") {
    tctx.fillStyle = "#000";
    tctx.beginPath(); tctx.ellipse(px + 8, py + 10, 6, 5, 0, Math.PI, 0); tctx.fill();
  } else if (ch === "," || ch === '"' || ch === ".") {
    if (r(5) < 0.4) {
      tctx.fillStyle = shadeHex(base, ch === "." ? 0.88 : 1.15);
      tctx.fillRect(px + r(6) * 13, py + r(8) * 13, 2, 2);
    }
  }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  shade.width = canvas.width; shade.height = canvas.height;
}
function fit() {
  const r = canvas.getBoundingClientRect();
  const span = S.size * 0.86;
  cam.scale = Math.min(r.width, r.height) / span;
  cam.x = S.size / 2 - r.width / cam.scale / 2;
  cam.y = S.size / 2 - r.height / cam.scale / 2 + 2;
}
const toScreen = (x, y) => [(x - cam.x) * cam.scale, (y - cam.y) * cam.scale];
const toWorld = (sx, sy) => [sx / cam.scale + cam.x, sy / cam.scale + cam.y];

function displayPos(a, now) {
  const t = Math.min(1, (now - S.tickAt) / (S.tickSeconds * 1000 * 0.85));
  return { x: a.fx + (a.x - a.fx) * t, y: a.fy + (a.y - a.fy) * t };
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------
function darkness() {
  const t = S.tod, start = 1 - S.nightFraction;
  if (t >= start) return 0.62;
  if (t >= start - 0.08) return 0.62 * (t - (start - 0.08)) / 0.08;
  if (t < 0.06) return 0.62 * (1 - t / 0.06);
  return 0;
}

function frame(now) {
  requestAnimationFrame(frame);
  if (!S.size) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width / dpr, H = canvas.height / dpr;
  if (S.follow && S.selected && S.agents[S.selected]) {
    const p = displayPos(S.agents[S.selected], now);
    cam.x += (p.x - W / cam.scale / 2 - cam.x) * 0.12;
    cam.y += (p.y - H / cam.scale / 2 - cam.y) * 0.12;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = TERRAIN["~"];
  ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = cam.scale < TILE;
  const [ox, oy] = toScreen(0, 0);
  ctx.drawImage(terrain, ox, oy, S.size * cam.scale, S.size * cam.scale);

  const z = cam.scale;
  drawBushes(z);
  drawStructures(z, now);
  drawLandmarks(z);

  // Night: darken everything, then cut holes where fires burn.
  const dark = darkness();
  if (dark > 0) {
    sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sctx.globalCompositeOperation = "source-over";
    sctx.clearRect(0, 0, W, H);
    sctx.fillStyle = `rgba(6, 12, 32, ${dark})`;
    sctx.fillRect(0, 0, W, H);
    sctx.globalCompositeOperation = "destination-out";
    for (const s of S.structures) {
      if (!s.burning) continue;
      const [sx, sy] = toScreen(s.x + 0.5, s.y + 0.5);
      const flick = 1 + Math.sin(now / 130 + s.id) * 0.03;
      const g = sctx.createRadialGradient(sx, sy, 0, sx, sy, 4.5 * z * flick);
      g.addColorStop(0, "rgba(0,0,0,1)"); g.addColorStop(0.6, "rgba(0,0,0,0.75)"); g.addColorStop(1, "rgba(0,0,0,0)");
      sctx.fillStyle = g;
      sctx.fillRect(sx - 5 * z, sy - 5 * z, 10 * z, 10 * z);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(shade, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalCompositeOperation = "lighter";
    for (const s of S.structures) {
      if (!s.burning) continue;
      const [sx, sy] = toScreen(s.x + 0.5, s.y + 0.5);
      const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, 3 * z);
      g.addColorStop(0, `rgba(255,140,40,${0.35 * dark})`); g.addColorStop(1, "rgba(255,140,40,0)");
      ctx.fillStyle = g;
      ctx.fillRect(sx - 3 * z, sy - 3 * z, 6 * z, 6 * z);
    }
    ctx.globalCompositeOperation = "source-over";
  }

  drawMonsters(z, now);
  drawAgents(z, now);
}

function drawBushes(z) {
  if (z < 4) return;
  for (const b of S.bushes.values()) {
    const [sx, sy] = toScreen(b.x + 0.5, b.y + 0.5);
    ctx.fillStyle = "#2f5a26";
    ctx.beginPath(); ctx.arc(sx, sy, z * 0.34, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#d8354f";
    for (let i = 0; i < b.n; i++) {
      const a = i * 1.9 + b.x;
      ctx.beginPath(); ctx.arc(sx + Math.cos(a) * z * 0.18, sy + Math.sin(a) * z * 0.18, Math.max(1, z * 0.08), 0, Math.PI * 2); ctx.fill();
    }
  }
}

function drawStructures(z, now) {
  for (const s of S.structures) {
    const [sx, sy] = toScreen(s.x, s.y);
    const cx = sx + z / 2, cy = sy + z / 2;
    const owner = s.owner && S.meta[s.owner];
    ctx.globalAlpha = s.complete ? 1 : 0.55;
    if (s.kind === "fire") {
      ctx.fillStyle = "#4a3426";
      ctx.fillRect(sx + z * 0.2, sy + z * 0.62, z * 0.6, z * 0.18);
      if (s.burning) {
        const f = 0.85 + Math.sin(now / 90 + s.id) * 0.15;
        ctx.fillStyle = "#ff8a2a";
        ctx.beginPath(); ctx.moveTo(sx + z * 0.25, sy + z * 0.68);
        ctx.quadraticCurveTo(cx, sy + z * (0.68 - 0.6 * f), sx + z * 0.75, sy + z * 0.68); ctx.fill();
        ctx.fillStyle = "#ffd36b";
        ctx.beginPath(); ctx.moveTo(sx + z * 0.38, sy + z * 0.68);
        ctx.quadraticCurveTo(cx, sy + z * (0.68 - 0.35 * f), sx + z * 0.62, sy + z * 0.68); ctx.fill();
      }
    } else if (s.kind === "storage") {
      ctx.fillStyle = "#7a5534"; ctx.fillRect(sx + z * 0.15, sy + z * 0.25, z * 0.7, z * 0.55);
      ctx.fillStyle = "#5a3d24"; ctx.fillRect(sx + z * 0.15, sy + z * 0.25, z * 0.7, z * 0.12);
    } else if (s.kind === "shelter") {
      ctx.fillStyle = "#9b7148";
      ctx.beginPath(); ctx.moveTo(sx + z * 0.05, sy + z * 0.85); ctx.lineTo(cx, sy + z * 0.1); ctx.lineTo(sx + z * 0.95, sy + z * 0.85); ctx.fill();
      ctx.fillStyle = "#2a1d12"; ctx.fillRect(cx - z * 0.1, sy + z * 0.5, z * 0.2, z * 0.35);
    } else if (s.kind === "farm") {
      // Soil darkens with water; rows of plants grow with the crop; ripe crops turn gold.
      ctx.fillStyle = s.water > 0.3 ? "#4e3420" : s.water > 0 ? "#6b4a2b" : "#9a7a52";
      ctx.fillRect(sx + z * 0.06, sy + z * 0.06, z * 0.88, z * 0.88);
      const h = s.stock ? 0.7 : Math.max(0.06, s.growth * 0.65);
      ctx.fillStyle = s.stock ? "#e3b23c" : s.water > 0 ? "#7dbb4a" : "#a3a054";
      for (let i = 0; i < 3; i++) ctx.fillRect(sx + z * (0.15 + i * 0.26), sy + z * (0.85 - h), z * 0.14, z * h);
      if (!s.stock && s.water <= 0 && s.complete && z >= 8) {
        ctx.fillStyle = "#6fd3f3"; ctx.beginPath();
        ctx.arc(sx + z * 0.82, sy + z * 0.2, z * 0.1, 0, Math.PI * 2); ctx.fill();
      }
    } else if (s.kind === "wall") {
      ctx.fillStyle = "#8f8a84"; ctx.fillRect(sx, sy, z, z);
      ctx.strokeStyle = "#5f5a55"; ctx.lineWidth = 1; ctx.strokeRect(sx + 0.5, sy + 0.5, z - 1, z - 1);
    } else if (s.kind === "remains") {
      ctx.strokeStyle = "#e8e0d0"; ctx.lineWidth = Math.max(1, z * 0.1);
      ctx.beginPath();
      ctx.moveTo(sx + z * 0.25, sy + z * 0.25); ctx.lineTo(sx + z * 0.75, sy + z * 0.75);
      ctx.moveTo(sx + z * 0.75, sy + z * 0.25); ctx.lineTo(sx + z * 0.25, sy + z * 0.75); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (!s.complete && s.needs && z >= 6) {
      // A project still waiting on materials: dashed outline.
      ctx.strokeStyle = "#f2ecdc"; ctx.lineWidth = 1; ctx.setLineDash([3, 2]);
      ctx.strokeRect(sx + 0.5, sy + 0.5, z - 1, z - 1); ctx.setLineDash([]);
    } else if (!s.complete && z >= 6) {
      ctx.fillStyle = "rgba(0,0,0,0.5)"; ctx.fillRect(sx, sy + z - 3, z, 3);
      ctx.fillStyle = "#fff"; ctx.fillRect(sx, sy + z - 3, z * s.progress, 3);
    }
    if (owner && z >= 6 && s.kind !== "remains") {
      ctx.fillStyle = owner.color;
      ctx.fillRect(sx + z - Math.max(3, z * 0.2), sy, Math.max(3, z * 0.2), Math.max(3, z * 0.2));
    }
  }
}

function structureLabel(s) {
  const owner = s.owner && S.meta[s.owner] ? `${S.meta[s.owner].name}'s ` : "";
  if (s.kind === "remains") return `the body of ${S.meta[s.of]?.name || "someone"}`;
  if (!s.complete) return s.needs ? `${owner}${s.kind} project, needs ${s.needs}` : `${owner}${s.kind}, ${Math.round(s.progress * 100)}% built`;
  if (s.kind === "farm") return s.stock ? `${owner}farm, ${s.stock} crops ripe` :
    `${owner}farm, ${Math.round(s.growth * 100)}% grown, soil ${s.water <= 0 ? "dry" : s.water < 0.3 ? "drying" : "watered"}`;
  if (s.kind === "fire") return `${owner}fire, ${s.burning ? "burning" : "burnt out"}`;
  if (s.kind === "storage") return `${owner}storage, ${s.items} items`;
  return `${owner}${s.kind}`;
}

function drawLandmarks(z) {
  ctx.textAlign = "center";
  ctx.font = `italic 400 ${Math.max(12, Math.min(18, z * 1.3))}px Newsreader, Georgia, serif`;
  for (const lm of S.landmarks) {
    const [sx, sy] = toScreen(lm.x + 0.5, lm.y - (lm.kind === "area" ? 0 : 1.2));
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(10, 22, 30, 0.7)";
    ctx.fillStyle = lm.hidden ? "rgba(230,224,207,0.6)" : "rgba(240,234,215,0.92)";
    ctx.strokeText(lm.name, sx, sy);
    ctx.fillText(lm.name, sx, sy);
  }
}

function drawMonsters(z, now) {
  for (const m of S.monsters) {
    const t = Math.min(1, (now - S.tickAt) / (S.tickSeconds * 850));
    const x = m.fx + (m.x - m.fx) * t, y = m.fy + (m.y - m.fy) * t;
    const [cx, cy] = toScreen(x + 0.5, y + 0.5);
    const r = z * 0.48;
    ctx.fillStyle = "#1a0707";
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = (i / 10) * Math.PI * 2 + now / 900;
      const rr = i % 2 ? r * 0.6 : r;
      ctx.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr);
    }
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "#8a1c1c"; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = "#ff3b30";
    ctx.beginPath(); ctx.arc(cx - r * 0.25, cy - r * 0.1, Math.max(1.2, r * 0.12), 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(cx + r * 0.25, cy - r * 0.1, Math.max(1.2, r * 0.12), 0, Math.PI * 2); ctx.fill();
  }
}

function drawAgents(z, now) {
  const list = Object.values(S.agents).filter((a) => a.alive);
  const bubbles = [];
  for (const a of list) {
    const meta = S.meta[a.id]; if (!meta) continue;
    const p = displayPos(a, now);
    const [cx, cy] = toScreen(p.x + 0.5, p.y + 0.5);
    const r = Math.max(4, z * 0.42);
    const sel = a.id === S.selected;
    if (sel) {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, r + 4, 0, Math.PI * 2); ctx.stroke();
    }
    if (a.thinking) {
      const k = (now / 700) % 1;
      ctx.strokeStyle = `rgba(255,255,255,${0.55 * (1 - k)})`; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, r + 2 + k * 6, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = meta.color;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.65)"; ctx.lineWidth = 1.5; ctx.stroke();
    if (z >= 10) {
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.font = `600 ${Math.round(r * 1.1)}px Instrument Sans, system-ui`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(meta.name[0], cx, cy + 0.5);
      ctx.textBaseline = "alphabetic";
    }
    if (a.hp < 0.95) {
      ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(cx - r, cy + r + 2, r * 2, 3);
      ctx.fillStyle = a.hp > 0.5 ? "#7fbf6a" : a.hp > 0.25 ? "#f0a44b" : "#e5484d";
      ctx.fillRect(cx - r, cy + r + 2, r * 2 * a.hp, 3);
    }
    if (a.sleeping) {
      ctx.fillStyle = "#cfe3ff"; ctx.font = `italic ${Math.max(10, r)}px Newsreader, serif`; ctx.textAlign = "left";
      ctx.fillText("z", cx + r * 0.8, cy - r * 0.8 - Math.sin(now / 500) * 2);
    }
    if (z >= 7 || sel) {
      ctx.font = `500 ${Math.max(11, Math.min(13, z))}px Instrument Sans, system-ui`;
      ctx.textAlign = "center";
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(8,18,26,0.85)";
      ctx.strokeText(meta.name, cx, cy + r + 15);
      ctx.fillStyle = "#f2ecdc"; ctx.fillText(meta.name, cx, cy + r + 15);
    }
    if (a.bubble) bubbles.push({ a, meta, cx, cy: cy - r - 6, sel });
  }
  bubbles.sort((b1, b2) => b1.sel - b2.sel);
  for (const b of bubbles) drawBubble(b);
}

function drawBubble({ a, meta, cx, cy }) {
  const vol = a.bubble.volume;
  let text = a.bubble.text;
  if (text.length > 90) text = text.slice(0, 88) + "…";
  const size = vol === "shout" ? 14 : 13;
  ctx.font = `${vol === "whisper" ? "italic " : ""}${vol === "shout" ? 600 : 400} ${size}px Newsreader, Georgia, serif`;
  const words = text.split(" "), lines = [];
  let line = "";
  for (const w of words) {
    const test = line ? line + " " + w : w;
    if (ctx.measureText(test).width > 200 && line) { lines.push(line); line = w; } else line = test;
  }
  lines.push(line);
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16;
  const h = lines.length * (size + 3) + 10;
  const x = cx - w / 2, y = cy - h - 6;
  ctx.fillStyle = vol === "whisper" ? "rgba(235,230,215,0.82)" : "rgba(248,244,232,0.96)";
  ctx.strokeStyle = vol === "shout" ? "#e5484d" : meta.color;
  ctx.lineWidth = vol === "shout" ? 2 : 1.5;
  ctx.setLineDash(vol === "whisper" ? [3, 3] : []);
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 7);
  ctx.moveTo(cx - 5, y + h); ctx.lineTo(cx, y + h + 6); ctx.lineTo(cx + 5, y + h);
  ctx.fill(); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#1b1f23"; ctx.textAlign = "left";
  lines.forEach((l, i) => ctx.fillText(l, x + 8, y + 5 + (i + 1) * (size + 3) - 3));
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------
const wrap = $("map-wrap");
let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY, cx: cam.x, cy: cam.y, moved: false };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  const r = canvas.getBoundingClientRect();
  const [wx, wy] = toWorld(e.clientX - r.left, e.clientY - r.top);
  const tx = Math.floor(wx), ty = Math.floor(wy);
  const names = { "~": "sea", "-": "fishing waters", ".": "sand", ",": "grass", '"': "meadow", T: "forest", F: "dense forest", "^": "hills", M: "mountain", r: "rocky ground", w: "pond", o: "spring", C: "cave mouth" };
  let label = S.size && tx >= 0 && ty >= 0 && tx < S.size && ty < S.size ? `(${tx}, ${ty}) ${names[S.grid[ty][tx]] || ""}` : "";
  const st = label && S.structures.find((x) => x.x === tx && x.y === ty);
  if (st) label += `, ${structureLabel(st)}`;
  $("hover").textContent = label;
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; wrap.classList.add("dragging"); S.follow = false; }
  cam.x = drag.cx - dx / cam.scale; cam.y = drag.cy - dy / cam.scale;
});
canvas.addEventListener("pointerup", (e) => {
  wrap.classList.remove("dragging");
  if (drag && !drag.moved) {
    const r = canvas.getBoundingClientRect();
    const [wx, wy] = toWorld(e.clientX - r.left, e.clientY - r.top);
    let best = null, bd = 0.9;
    for (const a of Object.values(S.agents)) {
      if (!a.alive) continue;
      const p = displayPos(a, performance.now());
      const d = Math.hypot(p.x + 0.5 - wx, p.y + 0.5 - wy);
      if (d < bd) { best = a; bd = d; }
    }
    if (best) select(best.id);
  }
  drag = null;
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const [wx, wy] = toWorld(mx, my);
  cam.scale = Math.max(3, Math.min(48, cam.scale * Math.exp(-e.deltaY * 0.0015)));
  cam.x = wx - mx / cam.scale; cam.y = wy - my / cam.scale;
  $("map-wrap").querySelector(".map-hint").classList.add("gone");
}, { passive: false });
window.addEventListener("keydown", (e) => {
  if (e.target.closest("select, input, textarea")) return;
  if (e.code === "Space") { e.preventDefault(); togglePause(); }
  else if (e.key === "f" || e.key === "F") { if (S.selected) { S.follow = !S.follow; renderMind(); } }
  else if (e.key === "Escape") { S.follow = false; select(null); }
});
window.addEventListener("resize", resize);

function togglePause() { send({ type: "control", action: S.paused ? "resume" : "pause" }); }
$("btn-pause").onclick = togglePause;
$("btn-step").onclick = () => send({ type: "control", action: "step" });
$("speed").onchange = (e) => send({ type: "control", action: "speed", value: parseFloat(e.target.value) });

// ---------------------------------------------------------------------------
// Top bar
// ---------------------------------------------------------------------------
function updateBar(m) {
  $("clock-text").textContent = `Day ${m.day}, ${m.phase}${m.paused ? " (paused)" : ""}`;
  $("dial-fill").style.left = `${m.tod * 100}%`;
  $("dial-night").style.width = `${S.nightFraction * 100}%`;
  $("btn-pause").textContent = m.paused ? "Resume" : "Pause";
  const sel = $("speed");
  if (document.activeElement !== sel && m.tick_seconds) sel.value = String(m.tick_seconds);
  const b = m.brain;
  const brain = $("brain");
  brain.innerHTML =
    `${esc(b.backend)}: <b>${b.inflight}</b> thinking, <b>${b.queued}</b> waiting, ` +
    `<b>${b.latency}s</b> per thought, <b>${b.tps}</b> tok/s` +
    (b.errors ? `, <span class="err">${b.errors} errors</span>` : "");
  brain.title = b.last_error || `${b.decisions} decisions so far`;
}

// ---------------------------------------------------------------------------
// Roster
// ---------------------------------------------------------------------------
function buildRoster() {
  const el = $("roster");
  el.innerHTML = "";
  for (const meta of Object.values(S.meta)) {
    const b = document.createElement("button");
    b.className = "who"; b.dataset.id = meta.id;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<span class="dot" style="background:${meta.color}"></span><span class="nm">${esc(meta.name)}</span><span class="doing"></span>`;
    b.onclick = () => select(meta.id);
    el.appendChild(b);
  }
}
function updateRoster() {
  for (const b of $("roster").children) {
    const a = S.agents[b.dataset.id]; if (!a) continue;
    b.classList.toggle("dead", !a.alive);
    b.classList.toggle("thinking", a.thinking);
    b.setAttribute("aria-pressed", String(b.dataset.id === S.selected));
    b.querySelector(".doing").textContent = !a.alive ? "dead" : a.sleeping ? "asleep" : shortAct(a.act);
    b.title = a.act;
  }
}
function shortAct(s) { return (s || "").replace(/^(walking|going) to /, "→ ").slice(0, 22); }

function select(id) {
  S.selected = id;
  S.detail = null;
  document.documentElement.style.setProperty("--agent", id ? S.meta[id].color : "#a5afb6");
  send({ type: "inspect", id });
  if (S.tab !== "power") setTab("mind");
  else renderPower();
  updateRoster();
  renderMind();
}

// ---------------------------------------------------------------------------
// Mind panel
// ---------------------------------------------------------------------------
const TABS = ["mind", "power", "island"];
function setTab(which) {
  S.tab = which;
  for (const t of TABS) {
    $(`tab-${t}`).setAttribute("aria-selected", String(t === which));
    $(t).hidden = t !== which;
  }
  if (which === "power") renderPower();
}
for (const t of TABS) $(`tab-${t}`).onclick = () => setTab(t);

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const cap = (s) => s ? s[0].toUpperCase() + s.slice(1) : s;

function meter(v, color) { return `<div class="meter"><i style="width:${Math.round(v * 100)}%;background:${color}"></i></div>`; }
function needColor(v) { return v > 0.85 ? "var(--bad)" : v > 0.6 ? "var(--warn)" : "#6a8ba0"; }
function biBar(v) {
  const w = Math.abs(v) * 50;
  return `<div class="bi"><i style="left:${v >= 0 ? 50 : 50 - w}%;width:${w}%;background:${v >= 0 ? "var(--good)" : "var(--bad)"}"></i></div>`;
}
function uniBar(v, color) { return `<div class="bi uni"><i style="left:0;width:${v * 100}%;background:${color}"></i></div>`; }

const LOG_MARK = { think: "", say: "", heard: "", act: "→ ", done: "✓ ", fail: "✗ ", event: "", feel: "", note: "✎ ", sys: "", death: "", memory: "" };

function org(d) {
  const bits = [];
  if (d.group) bits.push(d.leads ? `Leads <b>${esc(d.group)}</b>` : `Member of <b>${esc(d.group)}</b>`);
  if (d.job) bits.push(`Works for ${esc(d.job.for)}: “${esc(d.job.task)}”`);
  if (d.crew && d.crew.length) bits.push(`Working for them: ${d.crew.map(esc).join(", ")}`);
  for (const c of d.changes || []) bits.push(`Changed their mind (tick ${c.t}): “${esc(c.now)}”`);
  return bits.length ? `<p class="org">${bits.join(" · ")}</p>` : "";
}

function renderMind() {
  const el = $("mind");
  const d = S.detail;
  if (!S.selected) { el.innerHTML = `<p class="empty">Pick a castaway above or on the map to see what they're thinking.</p>`; return; }
  if (!d) { el.innerHTML = `<p class="empty">Listening…</p>`; return; }

  // Keep open <details> and the history scroll position across re-renders.
  const open = new Set([...el.querySelectorAll("details[open]")].map((x) => x.dataset.k));
  const scroll = el.scrollTop;

  const traits = d.traits.map(cap).join(", ");
  const status = d.alive
    ? `${esc(cap(d.task))}${d.thinking ? ' <span class="thinking">thinking…</span>' : d.queued ? ' <span class="thinking">about to think</span>' : ""}`
    : `Died ${esc(d.death_cause)}.`;
  const inv = Object.entries(d.inventory).map(([k, v]) => `<span class="chip"><b>${v}</b> ${esc(k)}</span>`).join("") || `<span class="chip">Nothing</span>`;
  const notes = d.notepad.length ? `<ol class="notepad">${d.notepad.map((n) => `<li>${esc(n)}</li>`).join("")}</ol>` : `<p class="empty">Nothing written yet.</p>`;
  const rels = d.relations.length ? d.relations.map((r) => `
    <div class="rel">
      <div class="top"><span class="dot" style="background:${r.color}"></span><span class="name">${esc(r.name)}${r.alive ? "" : " (dead)"}</span>
        ${r.obeyed || r.refused ? `<span class="chip">obeyed ${r.obeyed}, refused ${r.refused}</span>` : ""}</div>
      ${r.hint ? `<div class="hint">${esc(cap(r.hint))}</div>` : ""}
      <div class="bars">
        <div>Liking${biBar(r.affinity)}</div><div>Trust${biBar(r.trust)}</div>
        <div>Fear${uniBar(r.fear, "#b58cff")}</div><div>Respect${uniBar(r.respect, "#e9c46a")}</div>
      </div>
    </div>`).join("") : `<p class="empty">Hasn't really met anyone yet.</p>`;
  const reqs = d.requests.length ? `<h3>Waiting on an answer</h3><ul class="log">${d.requests.map((r) => `<li><span class="t"></span><span class="heard">${esc(r.from)}: “${esc(r.task)}”</span></li>`).join("")}</ul>` : "";
  const hist = d.history.slice().reverse().map((h) => `<li><span class="t">${h.t}</span><span class="${h.k}">${LOG_MARK[h.k] ?? ""}${esc(h.text)}</span></li>`).join("");

  el.innerHTML = `
    <div class="id-line"><h2>${esc(d.name)}</h2>${d.ambition ? `<span class="amb">${esc(cap(d.ambition))}</span>` : ""}</div>
    <p class="meta">${esc(traits)}. At (${d.pos[0]}, ${d.pos[1]}), ${d.think_count} thoughts, ${d.tier} model.
      <button id="follow" aria-pressed="${S.follow}">${S.follow ? "Following" : "Follow"}</button></p>
    <p class="doing-now">${status}</p>
    ${org(d)}
    ${d.alive ? `<div class="voice">
      <p class="purpose">${esc(d.purpose)}</p>
      ${d.inner.map((l) => `<p class="inner">${esc(l)}</p>`).join("")}
    </div>` : ""}
    <div class="two">
      <div><h3>Body</h3><div class="meters">
        <span>Health</span>${meter(d.hp, d.hp > 0.5 ? "var(--good)" : d.hp > 0.25 ? "var(--warn)" : "var(--bad)")}
        <span>Thirst</span>${meter(d.needs.thirst, needColor(d.needs.thirst))}
        <span>Hunger</span>${meter(d.needs.hunger, needColor(d.needs.hunger))}
        <span>Tiredness</span>${meter(d.needs.fatigue, needColor(d.needs.fatigue))}
      </div></div>
      <div><h3>Heart</h3><div class="meters">
        <span>Joy</span>${meter(d.emotions.joy, "#e9c46a")}
        <span>Sadness</span>${meter(d.emotions.sadness, "#5e8fd6")}
        <span>Anger</span>${meter(d.emotions.anger, "#e5484d")}
        <span>Fear</span>${meter(d.emotions.fear, "#b58cff")}
      </div></div>
    </div>
    <h3>Carrying</h3><div class="chips">${inv}</div>
    <h3>Memories</h3>${d.memories.length ? `<ul class="memories">${d.memories.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>` : `<p class="empty">Hasn't stopped to think back yet.</p>`}
    <h3>Notepad</h3>${notes}
    ${reqs}
    <h3>People</h3><div class="rels">${rels}</div>
    <h3>What they said, heard and did</h3><ul class="log">${hist}</ul>
    <details data-k="prompt"><summary>Last prompt</summary><pre>${esc(d.last_prompt || "No thoughts yet.")}</pre></details>
    <details data-k="raw"><summary>Last reply from the model</summary><pre>${esc(d.last_raw || "No reply yet.")}</pre></details>
  `;
  for (const x of el.querySelectorAll("details")) if (open.has(x.dataset.k)) x.open = true;
  el.scrollTop = scroll;
  $("follow").onclick = () => { S.follow = !S.follow; renderMind(); };
}

// ---------------------------------------------------------------------------
// Power: the hierarchy graph
// ---------------------------------------------------------------------------
const EDGES = [
  { key: "obeys", label: "Obeys", color: "#e9c46a", on: true },
  { key: "fears", label: "Fears", color: "#b58cff", on: true },
  { key: "likes", label: "Likes", color: "#7fbf6a", on: false },
  { key: "hates", label: "Can't stand", color: "#e5484d", on: false },
];
const powerCanvas = $("power-graph"), pctx = powerCanvas.getContext("2d");
let powerLayout = [];

function buildEdgeToggles() {
  const el = $("edge-toggles");
  el.innerHTML = "";
  for (const e of EDGES) {
    const b = document.createElement("button");
    b.className = "edge-toggle";
    b.setAttribute("aria-pressed", String(e.on));
    b.innerHTML = `<i style="background:${e.color}"></i>${e.label}`;
    b.onclick = () => { e.on = !e.on; b.setAttribute("aria-pressed", String(e.on)); renderPower(); };
    el.appendChild(b);
  }
}

function powerEdges(g) {
  // Normalize everything to arrows "from -> to" meaning "from obeys/fears/likes to".
  const out = [];
  for (const e of g.edges) {
    if (e.obeyed > 0) out.push({ kind: "obeys", from: e.to, to: e.from, w: Math.min(5, 1 + e.obeyed) });
    if (e.fear > 0.3) out.push({ kind: "fears", from: e.from, to: e.to, w: 1 + e.fear * 3 });
    if (e.affinity > 0.3) out.push({ kind: "likes", from: e.from, to: e.to, w: 1 + e.affinity * 2 });
    if (e.affinity < -0.3) out.push({ kind: "hates", from: e.from, to: e.to, w: 1 - e.affinity * 2 });
  }
  const on = new Set(EDGES.filter((x) => x.on).map((x) => x.key));
  return out.filter((e) => on.has(e.kind));
}

function renderPower() {
  const g = S.social;
  if (!g || S.tab !== "power") return;
  const dpr = window.devicePixelRatio || 1;
  const W = powerCanvas.clientWidth || 380, H = 440;
  if (powerCanvas.width !== Math.round(W * dpr)) { powerCanvas.width = Math.round(W * dpr); powerCanvas.height = Math.round(H * dpr); }
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  pctx.clearRect(0, 0, W, H);

  const nodes = g.nodes.filter((n) => n.alive);
  const max = Math.max(1, ...nodes.map((n) => n.standing));
  const top = 34, bottom = 40;
  // Height by standing, then spread people who land on the same level across the width.
  const placed = nodes.map((n) => ({ ...n, y: top + (1 - n.standing / max) * (H - top - bottom) }))
    .sort((a, b) => a.y - b.y || a.id.localeCompare(b.id));
  const rows = [];
  for (const n of placed) {
    const row = rows[rows.length - 1];
    if (row && n.y - row[0].y < 42) row.push(n); else rows.push([n]);
  }
  for (const row of rows) {
    const y = row.reduce((s, n) => s + n.y, 0) / row.length;
    row.sort((a, b) => a.id.localeCompare(b.id));
    row.forEach((n, i) => { n.x = (W * (i + 1)) / (row.length + 1); n.y = y; });
  }
  powerLayout = placed;
  const at = Object.fromEntries(placed.map((n) => [n.id, n]));
  const radius = (n) => 8 + Math.min(10, n.standing * 2);

  for (const e of powerEdges(g)) {
    const a = at[e.from], b = at[e.to];
    if (!a || !b) continue;
    const spec = EDGES.find((x) => x.key === e.kind);
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len, bend = 18;
    const cx = (a.x + b.x) / 2 + nx * bend, cy = (a.y + b.y) / 2 + ny * bend;
    // End the arrow at the target's edge.
    const tx = b.x - cx, ty = b.y - cy, tl = Math.hypot(tx, ty) || 1;
    const ex = b.x - (tx / tl) * (radius(b) + 3), ey = b.y - (ty / tl) * (radius(b) + 3);
    pctx.strokeStyle = spec.color; pctx.fillStyle = spec.color;
    pctx.globalAlpha = 0.8; pctx.lineWidth = e.w;
    pctx.setLineDash(e.kind === "hates" ? [4, 4] : []);
    pctx.beginPath(); pctx.moveTo(a.x, a.y); pctx.quadraticCurveTo(cx, cy, ex, ey); pctx.stroke();
    pctx.setLineDash([]);
    const ang = Math.atan2(ey - cy, ex - cx), h = 6 + e.w;
    pctx.beginPath();
    pctx.moveTo(ex, ey);
    pctx.lineTo(ex - h * Math.cos(ang - 0.45), ey - h * Math.sin(ang - 0.45));
    pctx.lineTo(ex - h * Math.cos(ang + 0.45), ey - h * Math.sin(ang + 0.45));
    pctx.fill();
    pctx.globalAlpha = 1;
  }
  for (const n of placed) {
    const r = radius(n);
    if (n.id === S.selected) { pctx.strokeStyle = "#fff"; pctx.lineWidth = 2; pctx.beginPath(); pctx.arc(n.x, n.y, r + 4, 0, Math.PI * 2); pctx.stroke(); }
    pctx.fillStyle = n.color; pctx.beginPath(); pctx.arc(n.x, n.y, r, 0, Math.PI * 2); pctx.fill();
    pctx.strokeStyle = "rgba(0,0,0,0.6)"; pctx.lineWidth = 1.5; pctx.stroke();
    pctx.font = "500 12px Instrument Sans, system-ui"; pctx.textAlign = "center";
    pctx.lineWidth = 3; pctx.strokeStyle = "#111d27"; pctx.strokeText(n.name, n.x, n.y + r + 14);
    pctx.fillStyle = "#e6e0cf"; pctx.fillText(n.name, n.x, n.y + r + 14);
  }
  pctx.fillStyle = "#6f7d88"; pctx.font = "12px Instrument Sans, system-ui"; pctx.textAlign = "left";
  pctx.fillText("Most influence", 4, 14);
  pctx.fillText("Least", 4, H - 6);

  const ranked = g.nodes.filter((n) => n.alive).sort((a, b) => b.standing - a.standing);
  $("ranking").innerHTML = ranked.map((n) => {
    const bits = [];
    if (n.group) bits.push(n.leads ? `leads ${esc(n.group)}` : `in ${esc(n.group)}`);
    if (n.crew) bits.push(`${n.crew} working for them`);
    if (n.works_for && S.meta[n.works_for]) bits.push(`works for ${esc(S.meta[n.works_for].name)}`);
    if (n.obeyed_by) bits.push(`obeyed by ${n.obeyed_by}`);
    if (n.feared_by) bits.push(`feared by ${n.feared_by}`);
    bits.push(n.liked >= 0.5 ? "well liked" : n.liked <= -0.5 ? "disliked" : "no strong feelings");
    return `<li><button class="lnk" data-id="${n.id}" style="color:${n.color}">${esc(n.name)}</button>` +
      `${n.ambition ? ` <span class="amb">${esc(n.ambition)}</span>` : ""}<br><span class="sub">${bits.join(", ")}</span></li>`;
  }).join("");
  for (const b of $("ranking").querySelectorAll("button.lnk")) b.onclick = () => select(b.dataset.id);
}

powerCanvas.addEventListener("click", (e) => {
  const r = powerCanvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  const hit = powerLayout.find((n) => Math.hypot(n.x - x, n.y - y) < 20);
  if (hit) select(hit.id);
});
buildEdgeToggles();

// ---------------------------------------------------------------------------
// Island feed
// ---------------------------------------------------------------------------
function renderIsland() {
  const el = $("island");
  const items = S.feed.slice(-200).reverse().map((f) => {
    const meta = f.actor && S.meta[f.actor];
    let text = esc(f.text);
    if (meta) text = text.replace(esc(meta.name), `<button class="lnk" data-id="${meta.id}" style="color:${meta.color}">${esc(meta.name)}</button>`);
    return `<li><span class="t">${f.t}</span><span class="${f.k}">${text}</span></li>`;
  }).join("");
  el.innerHTML = items ? `<ul class="feed">${items}</ul>` : `<p class="empty">Nothing has happened yet. Give them a minute.</p>`;
  for (const b of el.querySelectorAll("button.lnk")) b.onclick = () => select(b.dataset.id);
}

// ---------------------------------------------------------------------------
resize();
connect();
requestAnimationFrame(frame);
setTimeout(() => document.querySelector(".map-hint")?.classList.add("gone"), 9000);
