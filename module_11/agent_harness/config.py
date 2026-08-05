"""
config.py
=========
Single source of truth for provider configuration. Every module in this
course repeated these lines at the top of the file -- packaging them here
means changing settings (model, pricing, options) now happens in exactly
ONE place for your whole project.

This uses Groq's native Python client, which supports the same tool-calling
and chat completions conventions as the OpenAI API.
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
# (the openai/gpt-oss-* family). Leave LLM_OPTIONS empty for other models
# to avoid an API error.
LLM_OPTIONS = {"reasoning_effort": "none"} if MODEL.startswith("openai/gpt-oss") else {}

# A rough per-1K-token price table, used by cost.py for estimated cost
# tracking. These are illustrative placeholder numbers -- always check
# Groq's actual current pricing page (https://groq.com/pricing) for real
# values, since they vary per model.
PRICE_PER_1K_TOKENS = {
    "input": 0.00015,
    "output": 0.0006,
}

_client = None


def get_client() -> Groq:
    """
    Returns a single shared client instance (simple singleton pattern) so
    the whole application reuses one HTTP connection pool instead of
    creating a new client every time an agent is instantiated.
    """
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not found. Create a .env file with:\n"
                "  GROQ_API_KEY=gsk_...\n"
                "or export it directly in your shell."
            )
        _client = Groq(api_key=API_KEY)
    return _client