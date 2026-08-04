# Module 10 Extended — Notes

`module_10_extended.py` builds on Module 10 (multi-agent orchestration) and
Module 9 (tracing) to demonstrate three upgrades at once.

## 1. A third specialist: the fact-checker

The orchestrator originally only had two tools disguised as sub-agents:
`delegate_to_researcher` and `delegate_to_writer`. We added a third,
`delegate_to_fact_checker`, using the exact same pattern:

- its own system prompt ("review these notes for errors, don't rewrite them")
- its own tool schema (just `calculate`, to verify numbers — no web search,
  since it should scrutinize what it's given rather than go looking for
  new information)
- one more entry in `ORCHESTRATOR_TOOLS_SCHEMA` and `ORCHESTRATOR_TOOL_REGISTRY`

Nothing else about the orchestrator changed. That's the core idea behind
"sub-agents as tools" — adding a whole new specialist is just adding a new
tool description, not new orchestration logic. We also updated the
orchestrator's prompt to specify the order: research → fact-check → write.

## 2. Different models per specialist

`run_tool_calling_loop()` now takes a `model` argument instead of using one
global model for every call. Each sub-agent picks its own:

| Agent         | Model                         | Why                                  |
|---------------|--------------------------------|---------------------------------------|
| Orchestrator  | `llama-3.3-70b-versatile`     | General-purpose coordination          |
| Researcher    | `llama-3.1-8b-instant`        | Cheap/fast — may issue many tool calls|
| Fact-checker  | `llama-3.3-70b-versatile`     | Needs to reason carefully             |
| Writer        | `openai/gpt-oss-120b`         | Output quality matters most here      |

This required no structural changes because each sub-agent loop already ran
independently (its own `messages` list, own turn limit, own tools) — model
choice was just another parameter to thread through.

## 3. Per-agent tracing

Each `delegate_to_...` function now creates its **own** `Tracer` instance
(from Module 9), passes it into `run_tool_calling_loop()`, and saves it to
its own file:

```
traces/trace_orchestrator.json
traces/trace_researcher.json
traces/trace_fact_checker.json
traces/trace_writer.json
```

At the end, `print_combined_summary()` reads back `total_duration_ms()` and
`error_count()` from each agent's tracer and prints a one-line-per-agent
table — so you can immediately see, e.g., whether the cheap researcher
model was actually faster, or whether the fact-checker hit any tool errors,
without digging through one giant merged trace.

## How to run it

1. Put your Groq key in a `.env` file next to the script:
   ```
   GROQ_API_KEY=gsk_...
   ```
2. `python module_10_extended.py`
3. Inspect the per-agent trace files individually, or compare two of them
   the same way as before:
   ```python
   import json
   researcher = json.load(open("traces/trace_researcher.json"))
   writer = json.load(open("traces/trace_writer.json"))
   print(sum(e["duration_ms"] for e in researcher))
   print(sum(e["duration_ms"] for e in writer))
   ```