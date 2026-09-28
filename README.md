# LLMWorld

A text-based island sandbox. Twelve castaways, each driven by a small LLM, wash up on an
island with too little water, too few berries, and monsters that come out at night. They
talk, gather, build, steal, fight, and — ideally — sort themselves into hierarchies.

## Run it

```bash
uv sync
uv run python -m llmworld --scripted        # no LLM: simple scripted brains, good for testing
uv run python -m llmworld                   # uses config.yaml (Ollama by default)
```

Open <http://127.0.0.1:8765>. Drag to pan, scroll to zoom, click a castaway to see their
mind. `Space` pauses, `F` follows the selected castaway, `Esc` lets go.

The side panel has three tabs:

- **Mind** — the selected castaway: the inner voice their model receives, body and heart,
  memories, notepad, relationships, a full history, and the raw last prompt and reply.
- **Power** — the hierarchy as a graph. Height is influence (how much the others respect
  and fear them); arrows show who obeys, fears, likes, or can't stand whom.
- **Island** — everything public that happened, newest first.

### Choosing a model

```bash
uv run python scripts/bench.py qwen3:4b qwen3:8b            # same real prompts to each model
uv run python scripts/bench.py -n 60 --parallel 4 qwen3:8b  # match your server's parallel slots
```

Reports valid-JSON rate, bad actions (e.g. `gather` with no item, reacting to someone
who didn't speak), latency, throughput, and how often agents speak, make requests and
take notes, plus side-by-side samples. Full results go to `runs/bench-*.json`.

### On the RTX 3070 (8 GB)

One model at a time fits comfortably; set both tiers to the same model. Either:

**Ollama**

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
ollama pull qwen3:8b          # or qwen3:4b for roughly twice the speed
uv run python -m llmworld --main-model qwen3:8b --fast-model qwen3:8b --parallel 4
```

**llama-server** (better batching, so more total throughput)

```bash
llama-server -m Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 16384 --parallel 4 -fa -ctk q8_0 -ctv q8_0 --port 8080
uv run python -m llmworld --backend openai --url http://localhost:8080 --parallel 4
```

`-c` is shared across the parallel slots (16384 / 4 = 4096 tokens each). A turn is about
1.1k tokens of system prompt, 0.5k of turn prompt, and up to 400 out. Quantized KV cache
(`-ctk/-ctv q8_0`) keeps an 8B model with four slots inside 8 GB.

If the GPU is on another machine, run with `--host 0.0.0.0` there or point `--url` at it.

## How it works

```
┌──────────────┐  prompts built   ┌────────────────┐   HTTP, N parallel  ┌──────────────┐
│  Simulation  │ ───────────────▶ │  Brain queue   │ ──────────────────▶ │  LLM server  │
│ (tick = 2 s) │ ◀─────────────── │ (by priority)  │ ◀────────────────── │              │
└──────┬───────┘   decisions      └────────────────┘                     └──────────────┘
       │ JSON each tick over WebSocket
       ▼
   Web UI (map + mind inspector)
```

- **The sim never waits on the model.** It ticks every 2 s. Agents keep doing their current
  task (walking, gathering, building) while their next thought is generated; decisions are
  applied on the next tick.
- **Agents think when something happens:** spoken to, attacked, a task finished, a need got
  urgent, a monster appeared - or on a heartbeat every 8 ticks. Being addressed directly
  jumps the queue; overheard chatter only prompts a new thought after a short cooldown, so a
  crowd talking doesn't flood the model.
- **Prompt caching:** the system prompt starts with the rules everyone shares (~70% of it)
  and ends with who this agent is, so the server reuses one cached prefix across agents.
- **Actions are intents.** The model says `gather wood` or `move_to Central Spring`; the sim
  finds the nearest tree and pathfinds there.
- **One constrained JSON reply per thought:** reactions to what was heard, answers to
  requests, a private thought, an action, speech, a request of someone else, and a notepad edit.

### Minds

- **Inner voice, never labels.** Traits, ambitions, emotions and needs reach the prompt only
  as the agent's own thoughts: "Tom insulted you. Your fists are clenched. You want Tom to
  pay for it." — not "anger: 0.8".
- **Emotions** (joy, sadness, anger, fear) are numbers in the sim that events push around
  and that decay toward a personality baseline. Each keeps its *cause*, so the prose can say
  who and why.
- **Listeners judge speech.** Whether something was an insult is the listener's call: each
  reply includes how the words they heard made them feel (`insulted`, `threatened`,
  `flattered`, `persuaded`, ...). Fixed effects per feeling keep the sim predictable.
- **Relationships** per pair: liking, trust, fear, respect, plus how often each person
  obeyed or refused the other's requests. Hierarchy emerges from these.
- **Ambitions** (ruler, utopian, prophet, hoarder, needs-to-be-loved) sit in the system
  prompt as conviction and are repeated every turn, rebuilt from live facts: "Refused you:
  Tom. That cannot stand."
- **Notepad:** ten short lines each agent writes and always sees.
- **Long-term memory:** after about 30 new experiences, an agent "thinks back" (a
  low-priority model call) and rewrites up to 8 lines of what matters — grudges,
  alliances, debts, places. Those lines go into every prompt. Their own recent words are
  in the prompt too, so conversations keep their thread.

### The island

Built to force contact and conflict:

| Place | Why it matters |
|---|---|
| Central Spring, on The Pass | The water everyone knows about, on high ground that sees far, in the one easy gap through the ridge |
| Hidden Pond | The only other water, buried in dense forest near the caves; found only by stumbling onto it |
| Whispering Woods (west) | Wood everywhere, little food |
| Berry Meadows (east) | Food everywhere, no wood, no water |
| Shipwreck Beach, Fishing Lagoon | Where everyone starts; fish, but no fresh water |
| The Neck, South Point | A two-tile causeway to a peninsula: a fortress for whoever claims it |
| The Caves | Monsters come out at night; more each night |
| Eagle Rock, North Quarry | Lookout hill; stone |

Hills extend how far you can see; dense forest blocks sight; night shrinks it, and fires
light it back up.

## Tuning

- `config.yaml` — tick speed, day length, need rates, monsters, model settings.
- `data/agents.yaml` — the cast: names, traits, ambitions, backstories, model tier.
- `data/traits.yaml` — personalities as inner voice, plus how they bend emotions and trust.
- `data/ambitions.yaml` — drives and their live reminders.
- `data/phrases.yaml` — every emotion and need, as the agent would feel it.

Every run writes `runs/<timestamp>/events.jsonl` with everything that happened.
