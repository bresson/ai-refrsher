"""Real run — requires a live API key set as an environment variable.

    export OPENAI_API_KEY=sk-...
    python main.py

Only run this AFTER test_agent.py passes. This one costs real API
credits and needs a working key; test_agent.py does not.

DONT FORGET   phoenix serve
for local arize phoneix
"""
import json
from agent import run_agent
from tools import TOOLS, TOOL_FUNCTIONS
from observability import setup_tracing, traced_tools

setup_tracing()
TOOL_FUNCTIONS = traced_tools(TOOL_FUNCTIONS)

# Swap this one line to change providers later — nothing else in this
# project needs to change. Examples: "gpt-4o", "anthropic/claude-sonnet-4-5",
# "gemini/gemini-2.0-flash". Set the matching *_API_KEY env var for whichever
# you pick.
MODEL = "gpt-5.6-luna"

QUESTION_INDEX = 0  # change this to try a different question

if __name__ == "__main__":
    with open("questions.json") as f:
        questions = json.load(f)

    item = questions[QUESTION_INDEX]
    print("TASK_ID:", item["task_id"])
    print("QUESTION:", item["question"])

    answer = run_agent(MODEL, item["question"], TOOLS, TOOL_FUNCTIONS)
    print("ANSWER:", answer)