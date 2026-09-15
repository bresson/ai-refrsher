"""Tool implementations and schemas for this problem.

This is the only file you edit when moving to a new problem
(GAIA, Aviary, whatever comes next) — agent.py stays untouched.
"""
import logging
import os
from opentelemetry import trace

from tavily import TavilyClient

client = TavilyClient()  # reads TAVILY_API_KEY from env automatically

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("web_search")

MAX_CALLS = int(os.environ.get("TAVILY_MAX_CALLS", 900))  # stay under 1,000/month free tier
_call_count = 0

tracer = trace.get_tracer("web_search")

def web_search(query: str) -> str:
    global _call_count

    with tracer.start_as_current_span("web_search") as span:
        span.set_attribute("input.value", query)

        if _call_count >= MAX_CALLS:
            span.set_attribute("output.value", "budget_exhausted")
            return "[SEARCH_FAILED] Call budget exhausted for this session."

        _call_count += 1

        try:
            result = client.search(query, max_results=3, timeout=15)
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("output.value", f"error: {e}")
            return f"[SEARCH_FAILED] {e}"

        hits = result.get("results", [])
        formatted = "\n".join(f"{r['title']}: {r['content']}" for r in hits) if hits else "[SEARCH_EMPTY] No results found."
        span.set_attribute("output.value", formatted)
        return formatted


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call this when you have the final answer to the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The final answer."}
                },
                "required": ["answer"],
            },
        },
    },
]

TOOL_FUNCTIONS = {"web_search": web_search}