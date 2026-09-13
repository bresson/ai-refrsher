"""Tool implementations and schemas for this problem.

This is the only file you edit when moving to a new problem
(GAIA, Aviary, whatever comes next) — agent.py stays untouched.
"""


def web_search(query: str) -> str:
    """Placeholder search tool. Replace this body with a real
    implementation (a web_search API call, etc.) before using it
    for anything real. Kept as a stub here so the whole pipeline
    is runnable and testable with zero external dependencies.
    """
    return f"[PLACEHOLDER] No real search performed for: {query!r}"


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