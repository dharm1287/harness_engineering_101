"""
Module 10 Extended: Three Specialists, Per-Agent Models, Per-Agent Tracing
=============================================================================
Builds on Module 10 (Multi-Agent & Orchestration) and Module 9
(Observability & Debugging) to demonstrate three things at once:

  1. A THIRD SPECIALIST: a fact-checker agent, wired in as another
     `delegate_to_...` tool for the orchestrator. From the orchestrator's
     point of view this is no different from adding any other tool --
     that's the whole point of the "sub-agents as tools" pattern.

  2. PER-AGENT MODELS: each sub-agent (researcher, fact-checker, writer)
     can use a DIFFERENT model. Because each sub-agent's loop is fully
     independent (its own messages list, own turn limit, own tools), this
     requires no structural changes -- we just pass a different `model`
     string into each call to run_tool_calling_loop().

     In practice you'd pick a cheap/fast model for high-volume, simple
     work (like the researcher, which may loop over many searches) and a
     stronger model for the work that most needs quality (like the writer,
     whose output the user actually reads). Swap MODEL_RESEARCHER /
     MODEL_FACT_CHECKER / MODEL_WRITER below to whatever Groq models you
     have available.

  3. PER-AGENT TRACING: each sub-agent's `run_tool_calling_loop()` call
     gets its OWN Tracer instance, saved to its own file. This gives you
     an attributable trace per specialist -- you can answer "how long did
     the researcher spend?" or "did the fact-checker hit any errors?"
     without wading through one giant merged trace. We also print a combined
     summary at the end showing time spent per agent.

This uses Groq's native Python client, which supports the same tool-calling
and chat completions conventions as the OpenAI API.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from a .env file in the current directory
# (create one with a line like: GROQ_API_KEY=gsk_...)
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Get an API key at https://console.groq.com/keys and put it in a .env file:
#   GROQ_API_KEY=gsk_...
API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=API_KEY)

# --- PER-AGENT MODELS -------------------------------------------------------
# This is the new bit for requirement #2: each specialist gets its own model.
# Feel free to swap these for whatever Groq models you have access to, e.g.:
#   "llama-3.1-8b-instant"      -- fast/cheap, good for high-volume research
#   "llama-3.3-70b-versatile"   -- solid general-purpose default
#   "openai/gpt-oss-120b"       -- currently the most reliable Groq model
#                                   for tool calling; also strong at prose
MODEL_ORCHESTRATOR = "llama-3.3-70b-versatile"
MODEL_RESEARCHER = "llama-3.1-8b-instant"        # cheap/fast: lots of tool calls
MODEL_FACT_CHECKER = "llama-3.3-70b-versatile"   # needs to reason carefully
MODEL_WRITER = "openai/gpt-oss-120b"             # quality matters most here

# `reasoning_effort` is only supported by a subset of Groq models
# (the openai/gpt-oss-* family). We compute this per-call based on
# whichever model that specific call is using.
def llm_options_for(model: str) -> dict:
    return {"reasoning_effort": "none"} if model.startswith("openai/gpt-oss") else {}

MAX_TURNS_PER_AGENT = 6
TRACE_DIR = Path(__file__).resolve().parent / "traces"
TRACE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# TRACER (from Module 9, unmodified) -- collects structured events for ONE
# agent run and can save them to their own JSON file.
# ---------------------------------------------------------------------------
class Tracer:
    def __init__(self, run_id: str = None):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.events = []

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record(self, event_type: str, step: int, input_data, output_data, duration_ms: float, **extra):
        event = {
            "run_id": self.run_id,
            "step": step,
            "type": event_type,
            "timestamp": self._now(),
            "input": input_data,
            "output": output_data,
            "duration_ms": round(duration_ms, 1),
            **extra,
        }
        self.events.append(event)
        print(f"    [TRACE] step={step} type={event_type} duration={event['duration_ms']}ms")

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)
        print(f"    [TRACE] Saved {len(self.events)} events to '{path}'")

    def total_duration_ms(self) -> float:
        return sum(e["duration_ms"] for e in self.events)

    def error_count(self) -> int:
        return sum(1 for e in self.events if e.get("is_error"))


# ---------------------------------------------------------------------------
# GENERIC REUSABLE TOOL-CALLING LOOP (Module 3/10's shape), now extended to:
#   - accept a `model` argument, so each caller can pick its own model
#     (requirement #2)
#   - accept an optional `tracer`, so each caller can record structured
#     events for its own run instead of just printing (requirement #3)
# ---------------------------------------------------------------------------
def run_tool_calling_loop(system_prompt: str, user_message: str, tools_schema: list,
                            tool_registry: dict, agent_label: str, model: str,
                            tracer: "Tracer | None" = None) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    llm_options = llm_options_for(model)

    for turn in range(1, MAX_TURNS_PER_AGENT + 1):
        print(f"    [{agent_label} | turn {turn}] calling {model} with {len(messages)} messages...")

        start = time.monotonic()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema if tools_schema else None,
            **llm_options,
        )
        duration_ms = (time.monotonic() - start) * 1000
        message = response.choices[0].message

        if tracer:
            tracer.record(
                event_type="llm_call",
                step=turn,
                input_data=f"{len(messages)} messages (model={model})",
                output_data=(
                    f"tool_calls={[tc.function.name for tc in message.tool_calls]}"
                    if message.tool_calls else f"text={(message.content or '')[:150]}"
                ),
                duration_ms=duration_ms,
            )

        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_fn = tool_registry.get(tool_name)

                print(f"    [{agent_label} | turn {turn}] -> {tool_name}({tool_args})")

                tool_start = time.monotonic()
                if tool_fn is None:
                    tool_result = f"ERROR: unknown tool '{tool_name}'"
                else:
                    try:
                        tool_result = str(tool_fn(**tool_args))
                    except Exception as e:
                        tool_result = f"ERROR: {e}"
                tool_duration_ms = (time.monotonic() - tool_start) * 1000

                print(f"    [{agent_label} | turn {turn}] <- {tool_result[:100]}")

                if tracer:
                    tracer.record(
                        event_type="tool_call",
                        step=turn,
                        input_data={"tool": tool_name, "args": tool_args},
                        output_data=tool_result,
                        duration_ms=tool_duration_ms,
                        is_error=tool_result.startswith("ERROR:"),
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })
            continue

        print(f"    [{agent_label}] finished after {turn} turn(s).")
        return message.content

    return f"({agent_label} did not finish within its turn limit.)"


# ---------------------------------------------------------------------------
# SUB-AGENT 1: RESEARCHER
# Has real tools (calculate, search_web). Uses the cheap/fast model since
# it's likely to make several tool calls per run.
# ---------------------------------------------------------------------------
RESEARCHER_SYSTEM_PROMPT = """You are a research specialist. Use your tools
to gather facts and figures relevant to the question. Respond with concise,
factual notes -- NOT polished prose. Just the raw findings.
"""

RESEARCHER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for a query and return a short summary.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def _calculate(expression: str) -> str:
    return str(eval(expression, {"__builtins__": {}}, {}))


def _search_web(query: str) -> str:
    return f"[STUB RESULT] Pretend web search results for: '{query}'."


RESEARCHER_TOOL_REGISTRY = {"calculate": _calculate, "search_web": _search_web}


def delegate_to_researcher(question: str) -> str:
    """
    Exposed to the orchestrator as a "tool". Runs a complete, independent
    researcher agent loop with its own tracer and its own (cheap) model.
    """
    tracer = Tracer(run_id="researcher")
    result = run_tool_calling_loop(
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        user_message=question,
        tools_schema=RESEARCHER_TOOLS_SCHEMA,
        tool_registry=RESEARCHER_TOOL_REGISTRY,
        agent_label="RESEARCHER",
        model=MODEL_RESEARCHER,
        tracer=tracer,
    )
    tracer.save(TRACE_DIR / "trace_researcher.json")
    AGENT_TRACERS["researcher"] = tracer
    return result


# ---------------------------------------------------------------------------
# SUB-AGENT 2: FACT-CHECKER  (NEW -- requirement #1)
# Given a set of research notes, checks them for internal consistency and
# flags anything that looks unsupported or questionable. Has its own
# `calculate` tool (so it can double-check any arithmetic in the notes) but
# no search tool -- it's meant to scrutinize what it's given, not go
# looking for new information.
# ---------------------------------------------------------------------------
FACT_CHECKER_SYSTEM_PROMPT = """You are a fact-checking specialist. You will
be given a set of research notes. Carefully check them for internal
consistency, unsupported claims, or arithmetic errors (use your calculate
tool to verify any numeric claims). Respond with a short list of any issues
found, or state clearly "No issues found" if the notes look sound. Do not
rewrite the notes -- only flag problems.
"""

FACT_CHECKER_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression, used to verify numeric claims.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

FACT_CHECKER_TOOL_REGISTRY = {"calculate": _calculate}


def delegate_to_fact_checker(notes_to_check: str) -> str:
    """
    Exposed to the orchestrator as a "tool", same pattern as the researcher
    and writer. Runs its own independent agent loop, its own tracer, and
    its own model.
    """
    tracer = Tracer(run_id="fact_checker")
    result = run_tool_calling_loop(
        system_prompt=FACT_CHECKER_SYSTEM_PROMPT,
        user_message=notes_to_check,
        tools_schema=FACT_CHECKER_TOOLS_SCHEMA,
        tool_registry=FACT_CHECKER_TOOL_REGISTRY,
        agent_label="FACT-CHECKER",
        model=MODEL_FACT_CHECKER,
        tracer=tracer,
    )
    tracer.save(TRACE_DIR / "trace_fact_checker.json")
    AGENT_TRACERS["fact_checker"] = tracer
    return result


# ---------------------------------------------------------------------------
# SUB-AGENT 3: WRITER
# No tools at all -- pure text specialist. Uses the strongest model, since
# its output is what the end user actually reads.
# ---------------------------------------------------------------------------
WRITER_SYSTEM_PROMPT = """You are a writing specialist. You will be given
raw research notes, any fact-check feedback, and instructions. Turn them
into clear, polished, well-organized prose suitable for a non-technical
reader. Do not invent facts not present in the notes. If the fact-checker
flagged issues, make sure your final text does not repeat them.
"""


def delegate_to_writer(notes_and_instructions: str) -> str:
    """
    No tools_schema/tool_registry needed -- pure text-in/text-out loop.
    Still gets its own tracer and its own (stronger) model.
    """
    tracer = Tracer(run_id="writer")
    result = run_tool_calling_loop(
        system_prompt=WRITER_SYSTEM_PROMPT,
        user_message=notes_and_instructions,
        tools_schema=[],
        tool_registry={},
        agent_label="WRITER",
        model=MODEL_WRITER,
        tracer=tracer,
    )
    tracer.save(TRACE_DIR / "trace_writer.json")
    AGENT_TRACERS["writer"] = tracer
    return result


# Keeps a reference to each sub-agent's tracer after it runs, so we can
# print a combined summary at the very end without threading tracers
# through every function signature.
AGENT_TRACERS: dict = {}


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# Its "tools" are three whole sub-agents. From the orchestrator LLM's point
# of view this looks exactly like Module 3's tool calling -- adding the
# fact-checker was just adding one more entry to ORCHESTRATOR_TOOLS_SCHEMA
# and ORCHESTRATOR_TOOL_REGISTRY, no other structural change needed.
# ---------------------------------------------------------------------------
ORCHESTRATOR_SYSTEM_PROMPT = """You are an orchestrator managing three
specialists:
  - a researcher (gathers facts/figures)
  - a fact-checker (reviews research notes for errors or unsupported claims)
  - a writer (produces polished prose from notes)

Delegate in this order: research first, then have the fact-checker review
the research notes, then pass the notes (and any fact-check feedback) to
the writer. Give the final polished text as your own last message once the
writer has produced it.
"""

ORCHESTRATOR_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_researcher",
            "description": "Delegate a research question to the researcher specialist agent.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_fact_checker",
            "description": "Delegate a set of research notes to the fact-checker specialist agent for review.",
            "parameters": {
                "type": "object",
                "properties": {"notes_to_check": {"type": "string"}},
                "required": ["notes_to_check"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_to_writer",
            "description": "Delegate writing to the writer specialist, given research notes, fact-check feedback, and instructions.",
            "parameters": {
                "type": "object",
                "properties": {"notes_and_instructions": {"type": "string"}},
                "required": ["notes_and_instructions"],
            },
        },
    },
]

ORCHESTRATOR_TOOL_REGISTRY = {
    "delegate_to_researcher": delegate_to_researcher,
    "delegate_to_fact_checker": delegate_to_fact_checker,
    "delegate_to_writer": delegate_to_writer,
}


def run_orchestrator(user_task: str) -> str:
    print(f"\n{'=' * 60}")
    print("[ORCHESTRATOR] Starting top-level orchestration loop...")
    print(f"{'=' * 60}")
    tracer = Tracer(run_id="orchestrator")
    result = run_tool_calling_loop(
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        user_message=user_task,
        tools_schema=ORCHESTRATOR_TOOLS_SCHEMA,
        tool_registry=ORCHESTRATOR_TOOL_REGISTRY,
        agent_label="ORCHESTRATOR",
        model=MODEL_ORCHESTRATOR,
        tracer=tracer,
    )
    tracer.save(TRACE_DIR / "trace_orchestrator.json")
    AGENT_TRACERS["orchestrator"] = tracer
    return result


# ---------------------------------------------------------------------------
# COMBINED SUMMARY ACROSS ALL AGENTS (requirement #3, the payoff)
# ---------------------------------------------------------------------------
def print_combined_summary():
    """
    Prints a short table comparing time spent and errors across every
    sub-agent that ran this session, pulled from each agent's own Tracer.
    Note some agents (researcher, fact-checker, writer) may not appear if
    the orchestrator chose not to call them.
    """
    print(f"\n{'=' * 70}")
    print("COMBINED TRACE SUMMARY (per specialist)")
    print(f"{'=' * 70}")
    print(f"{'Agent':<15}{'Model':<28}{'Events':<9}{'Total ms':<12}{'Errors':<8}")
    print("-" * 70)

    model_by_agent = {
        "orchestrator": MODEL_ORCHESTRATOR,
        "researcher": MODEL_RESEARCHER,
        "fact_checker": MODEL_FACT_CHECKER,
        "writer": MODEL_WRITER,
    }

    for agent_name, tracer in AGENT_TRACERS.items():
        print(f"{agent_name:<15}{model_by_agent.get(agent_name, '?'):<28}"
              f"{len(tracer.events):<9}{tracer.total_duration_ms():<12.1f}{tracer.error_count():<8}")

    print("-" * 70)
    print(f"Individual trace files saved under: {TRACE_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file with:\n"
            "  GROQ_API_KEY=gsk_...\n"
            "or export it directly in your shell."
        )

    task = (
        "I need a short paragraph, for a general audience, explaining what "
        "an 'agent harness' is and why tool calling matters. Also mention "
        "how many days are in 6 weeks as a fun aside. Please research the "
        "topic first, have the notes fact-checked, then have it written up nicely."
    )
    print(f"USER TASK: {task}")

    result = run_orchestrator(task)

    print(f"\n{'#' * 60}")
    print("FINAL ANSWER:")
    print(result)
    print(f"{'#' * 60}")

    print_combined_summary()