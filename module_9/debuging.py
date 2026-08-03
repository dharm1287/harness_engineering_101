import json

with open("agent_trace.json") as f:
    trace = json.load(f)

print(len(trace), "events")
print(json.dumps(trace[0], indent=2))  

llm_calls = [e for e in trace if e["type"] == "llm_call"]

# Total time spent in tool calls vs LLM calls
tool_time = sum(e["duration_ms"] for e in trace if e["type"] == "tool_call")
llm_time = sum(e["duration_ms"] for e in trace if e["type"] == "llm_call")
print(f"LLM: {llm_time}ms, Tools: {tool_time}ms")

# Which specific tool was slowest?
tool_events = [e for e in trace if e["type"] == "tool_call"]
slowest = max(tool_events, key=lambda e: e["duration_ms"])
print("slowest tool call:", slowest)

errors = [e for e in trace if e.get("is_error")]
for e in errors:
    print("error:", e["input"], "->", e["output"])