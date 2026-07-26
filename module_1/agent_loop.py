"""
Module 1: Minimal Agent Loop
=============================
Goal: Go from a single LLM call (Module 0) to a multi-turn "agent loop"
that keeps thinking and responding until it decides it is done.

There are NO tools yet in this module. The only thing we add compared to
Module 0 is:
  1. A conversation history (list of messages) that persists across turns.
  2. A loop that keeps calling the LLM until some stopping condition is met.

This script uses Groq's native Python client to talk
to Groq-hosted models.
"""

import os
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
#
# Pick any currently supported Groq model, e.g.:
#   "llama-3.3-70b-versatile"
#   "llama-3.1-8b-instant"
#   "openai/gpt-oss-120b"
#   "openai/gpt-oss-20b"

API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"

# `reasoning_effort` is only supported by a subset of Groq models
# (e.g. the openai/gpt-oss-* family). Leave LLM_OPTIONS empty for
# models that don't support it to avoid an API error.
LLM_OPTIONS = {"reasoning_effort": "none"} if MODEL.startswith("openai/gpt-oss") else {}

# Create one client instance using Groq's native SDK.
client = Groq(api_key=API_KEY)

# A special phrase the model can output when it believes the task is complete.
# In later modules this will be replaced by a structured "tool call" signal,
# but for Module 1 (no tools yet) a plain text marker is simplest to reason about.
DONE_MARKER = "TASK_COMPLETE"

# System prompt: tells the model how the loop works so it can cooperate with it.
SYSTEM_PROMPT = f"""You are a helpful assistant operating inside an agent loop.
You will be given a task and can respond over multiple turns to think it through.
When you are fully done and ready to give your final answer, start your message
with the exact marker "{DONE_MARKER}" followed by your final answer.
If you are not yet done, just respond normally and you will be prompted to continue.
"""

# Safety valve: never loop forever. This cap exists purely so a buggy or
# confused model can't run up API costs indefinitely.
MAX_TURNS = 6


def run_agent_loop(user_task: str) -> str:
    """
    Runs the minimal agent loop for a single user task.

    The loop structure is the heart of Module 1:
        1. Send the full conversation history to the LLM.
        2. Print what we're about to do (for learning purposes).
        3. Get the LLM's response and print it.
        4. Check if the response contains the DONE_MARKER.
           - If yes: stop and return the final answer.
           - If no: append the response to history, nudge the model to
             continue, and go back to step 1.
    """

    # The conversation history is just a list of {role, content} dicts.
    # This is the "state" of our agent -- in later modules this will grow
    # to include tool calls and tool results too.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]

    for turn in range(1, MAX_TURNS + 1):
        # --- EDUCATIONAL LOGGING: show exactly what step we're on ---
        print(f"\n{'=' * 60}")
        print(f"[STEP {turn}] Sending {len(messages)} messages to the LLM...")
        print(f"{'=' * 60}")

        # Call the LLM. Note this is a completely standard Groq
        # chat.completions call -- nothing agent-specific about the API call
        # itself. The "agent" behavior comes entirely from the loop around it.
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            **LLM_OPTIONS,
        )
        assistant_reply = response.choices[0].message.content

        print(f"[STEP {turn}] LLM responded:")
        print(f"---\n{assistant_reply}\n---")

        # Record the assistant's reply in history so it has memory of what
        # it already said on the next iteration.
        messages.append({"role": "assistant", "content": assistant_reply})

        # --- Stopping condition check ---
        if DONE_MARKER in assistant_reply:
            final_answer = assistant_reply.split(DONE_MARKER, 1)[1].strip()
            print(f"\n[STEP {turn}] Done marker found. Ending loop.")
            return final_answer

        # Not done yet: nudge the model to keep going. This message plays
        # the same role a "tool result" will play in Module 2 -- it's new
        # information injected into the conversation to drive the next step.
        print(f"[STEP {turn}] No done marker yet. Continuing loop...")
        messages.append({
            "role": "user",
            "content": (
                f"Continue. If you are finished, remember to start your "
                f"reply with '{DONE_MARKER}'."
            ),
        })

    # If we hit MAX_TURNS without a done marker, we stop anyway. This is an
    # important lesson for real agents: always have a hard ceiling on loop
    # length so a confused model can't run forever.
    print("\n[WARNING] Max turns reached without a done marker.")
    return "(Agent did not finish within the turn limit.)"


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file with:\n"
            "  GROQ_API_KEY=gsk_...\n"
            "or export it directly in your shell."
        )

    # Try a task simple enough to reason about in 1-3 turns, so you can
    # clearly watch the loop mechanics without other complexity getting
    # in the way. Tool use comes in Module 2.
    task = "Explain, in exactly 3 short bullet points, why agent loops need a stopping condition."
    print(f"USER TASK: {task}")

    result = run_agent_loop(task)

    print(f"\n{'#' * 60}")
    print("FINAL ANSWER:")
    print(result)
    print(f"{'#' * 60}")