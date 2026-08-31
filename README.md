# agentic-ai-90

A 90-day, week-by-week journey into agentic AI — building up from raw LLM
HTTP calls to full tool-using agents. Each week lives in its own folder with
small, self-contained scripts that isolate one concept at a time.

Everything runs against a **local model served by [Ollama](https://ollama.com)**,
so there are no API keys and no per-token cost.

---

## Repository layout

```
agentic-ai-90/
└── week01/          # Talking to a local LLM over raw HTTP
    ├── hello_llm.py
    ├── hello_llm1.py
    ├── streaming.py
    └── extract_json.py
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running locally
  (`ollama serve` exposes the API on `http://localhost:11434`)
- A pulled chat model, e.g.:

  ```bash
  ollama pull qwen3:8b
  ```

  The scripts set the model name in a `MODEL` constant at the top of each
  file — change it to whatever `ollama list` shows on your machine.

## Setup

```bash
git clone https://github.com/<your-username>/agentic-ai-90.git
cd agentic-ai-90

python3 -m venv .venv
source .venv/bin/activate
pip install requests
```

Run any script directly:

```bash
python week01/streaming.py
```

---

## Week 01 — Hello, local LLM

**Goal:** understand what an LLM API call actually *is*, without any SDK or
framework in the way. Every script here uses nothing but `requests` and
`json` against Ollama's `/api/chat` endpoint.

| File | What it does |
|------|--------------|
| [hello_llm.py](week01/hello_llm.py) | The first end-to-end call. Sends a `messages` list to `/api/chat` with `stream: False`, gets one complete JSON response back, and prints the raw payload. |
| [hello_llm1.py](week01/hello_llm1.py) | A working copy / second pass over the same script, kept as a scratch variant while iterating. |
| [streaming.py](week01/streaming.py) | The same request with `stream: True`. The response arrives as newline-delimited JSON chunks; the loop reads them with `iter_lines()`, pulls `message.content` out of each one, and prints it token-by-token as it is generated. |
| [extract_json.py](week01/extract_json.py) | Placeholder for the next step — coaxing the model into returning **structured JSON** instead of prose, and parsing it reliably. Not implemented yet. |

**Main things covered this week:**

1. **The chat message format** — `role` / `content` pairs are the universal
   shape of an LLM request. Everything later (system prompts, tool calls,
   agent memory) is built on this list.
2. **Blocking vs. streaming responses** — `stream: False` gives one JSON
   object; `stream: True` gives a stream of JSON lines. Streaming is what
   makes an assistant feel responsive.
3. **Reading performance metrics off the response.** Ollama returns counters
   that `hello_llm.py` turns into a small report:
   - `prompt_eval_count` → prompt tokens (how much context you paid for)
   - `eval_count` → output tokens generated
   - `prompt_eval_duration` → time spent ingesting the prompt
   - `eval_duration` → time spent generating
   - `total_duration` → wall-clock time
   - and the derived **tokens/second** — the number to watch when comparing
     models or quantisations on your hardware.
4. **Durations are nanoseconds** — hence the `/ 1e9` conversions.
5. **Failing loudly** — `response.raise_for_status()` so a dead Ollama
   server or a bad model name surfaces immediately instead of producing a
   confusing `KeyError` further down.

> **Known issue:** `hello_llm.py` and `hello_llm1.py` currently have a stray
> `s` on the `"stream": False,s` line, which is a syntax error. Remove the
> trailing `s` before running them. `streaming.py` runs as-is.

---

## Roadmap

- [x] **Week 01** — raw HTTP calls to a local LLM, streaming, token metrics
- [ ] **Week 02** — structured output & JSON extraction
- [ ] **Week 03** — tool / function calling
- [ ] **Week 04** — the agent loop (plan → act → observe → repeat)

*(Weeks are added to this README as their folders land.)*
# Agentic-Ai
