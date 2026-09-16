"""Real run — requires a live API key set as an environment variable.

    export OPENAI_API_KEY=sk-...
    python main.py

Only run this AFTER test_agent.py passes. This one costs real API
credits and needs a working key; test_agent.py does not.

DONT FORGET   phoenix serve
for local arize phoneix
"""
import json
from pathlib import Path
from agent import run_agent
from tools import TOOLS, TOOL_FUNCTIONS
from observability import setup_tracing, traced_tools
from evaluation import submit_answers

setup_tracing()
TOOL_FUNCTIONS = traced_tools(TOOL_FUNCTIONS)

# Swap this one line to change providers later — nothing else in this
# project needs to change. Examples: "gpt-4o", "anthropic/claude-sonnet-4-5",
# "gemini/gemini-2.0-flash". Set the matching *_API_KEY env var for whichever
# you pick.
MODEL = "gpt-5.6-sol"

system_prompt = """
For any question, you must discern what data needs to be searched!
DO NOT USE TRAINING DATA in place of data that can be searched, looked up. 
Never answer from your own knowledge alone, even if you're confident.
If no applicable tool is available, state "No applicable tool"
"""

QUESTION_INDEX = 0  # change this to try a different question
USERNAME = "bresson"
AGENT_CODE = "https://github.com/bresson/ai-refrsher.git"
CACHE_PATH = Path("answer_cache.json")

if __name__ == "__main__":
    with open("questions.json") as f:
        questions = json.load(f)

        item = questions[QUESTION_INDEX]
        print("TASK_ID:", item["task_id"])
        print("QUESTION:", item["question"])

        answer = run_agent(MODEL, item["question"], TOOLS, TOOL_FUNCTIONS)
        print("ANSWER:", answer)

        result = submit_answers(
            username=USERNAME,
            agent_code=AGENT_CODE,
            answers_payload=[{"task_id": item["task_id"], "submitted_answer": answer}],
        )
        print(result)

        if result.get("correct_count", 0) > 0:
            cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
            cache[item["task_id"]] = answer
            CACHE_PATH.write_text(json.dumps(cache, indent=2))
        else:
            print(f"Incorrect. task_id={item['task_id']} answer={answer!r} result={result}")