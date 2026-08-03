import json

with open(r"C:\Users\dharm\Downloads\Harness_Engineering\module_9\agent_trace.json", "r") as f:
    trace = json.load(f)

[e for e in trace if e["type"] == "tool_call" and e["duration_ms"] > 500]

print("Tool calls with duration greater than 500 ms:")
for e in trace:
    print(e)