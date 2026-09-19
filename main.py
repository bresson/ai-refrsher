"""Real run — requires a live API key set as an environment variable.

    export OPENAI_API_KEY=sk-...
    python main.py

Only run this AFTER test_agent.py passes. This one costs real API
credits and needs a working key; test_agent.py does not.

DONT FORGET   phoenix serve
for local arize phoneix
"""
import os
import json
from pathlib import Path
import requests
from agent import agent_answer
from tools import TOOLS, TOOL_FUNCTIONS
from observability import setup_tracing, traced_tools
from evaluation import submit_answers

setup_tracing()
TOOL_FUNCTIONS = traced_tools(TOOL_FUNCTIONS)

# Swap this one line to change providers later — nothing else in this
# project needs to change. Examples: "gpt-4o", "anthropic/claude-sonnet-4-5",
# "gemini-3.6-flash". Set the matching *_API_KEY env var for whichever
# you pick.
MODEL = "gpt-5.6-sol"
VID_MODEL="gemini-3.6-flash"
TYPESAFE_API_KEY = os.environ["TYPESAFE_API_KEY"]
CONFIDENCE_THRESHOLD = 0.7 

system_prompt = """
For any question, you must discern what data needs to be searched!
DO NOT USE TRAINING DATA in place of data that can be searched, looked up. 
Never answer from your own knowledge alone, even if you're confident.
If no applicable tool is available, state "No applicable tool"
"""

QUESTION_INDEX = 2  # change this to try a different question
USERNAME = "bresson"
AGENT_CODE = "https://github.com/bresson/ai-refrsher.git"
CACHE_PATH = Path("answer_cache.json")

def judge_reasoning(question: str, answer: str, reasoning: str) -> float:
    resp = requests.post(
        "https://api.typesafe.ai/v1/systemone",
        headers={"Authorization": f"Bearer {TYPESAFE_API_KEY}"},
        json={
            "state": {"question": question, "answer": answer, "reasoning": reasoning},
            "model": "jev-latest",
            "questions": {
                "well_supported": {
                    "type": "noul",
                    "instructions": "Is the answer well-supported by the reasoning given?",
                    "criteria": {
                        "true": "The reasoning clearly justifies the answer",
                        "false": "The reasoning is vague, contradictory, or doesn't support the answer"
                    }
                }
            }
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["answers"]["well_supported"]["noul"]

if __name__ == "__main__":
    with open("questions.json") as f:
        questions = json.load(f)

        item = questions[QUESTION_INDEX]
        print("TASK_ID:", item["task_id"])
        print("QUESTION:", item["question"])

        answer, reasoning = agent_answer(item["question"], file_path=None)
        print("ANSWER:", answer)
        print('/n--------------------------------/n')
        print("REASONING:", reasoning)

        confidence = judge_reasoning(item["question"], answer, reasoning) if reasoning else None
        print("JEV_CONFIDENCE:", confidence)

        if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
            print(f"Jev flagged low confidence ({confidence:.2f}) — not submitting. task_id={item['task_id']}")
            # inert — automate retry later, per your note
        else:

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