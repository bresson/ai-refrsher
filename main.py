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
from evaluation import submit_answers, get_file_path

setup_tracing()
TOOL_FUNCTIONS = traced_tools(TOOL_FUNCTIONS)

# Swap this one line to change providers later — nothing else in this
# project needs to change. Examples: "gpt-4o", "anthropic/claude-sonnet-4-5",
# "gemini-3.6-flash". Set the matching *_API_KEY env var for whichever
# you pick.
MODEL = "gpt-5.6-sol"
VID_MODEL="claude-opus-5"
TYPESAFE_API_KEY = os.environ["TYPESAFE_API_KEY"]
CONFIDENCE_THRESHOLD = 0.7

system_prompt = """
For any question, you must discern what data needs to be searched!
DO NOT USE TRAINING DATA in place of data that can be searched, looked up. 
Never answer from your own knowledge alone, even if you're confident.
If no applicable tool is available, state "No applicable tool"
"""

QUESTION_INDEX = 3  # change this to try a different question
USERNAME = "bresson"
AGENT_CODE = "https://github.com/bresson/ai-refrsher.git"
CACHE_PATH = Path("answer_cache.json")

def judge_reasoning(question: str, answer: str, reasoning: str) -> tuple[float, str]:
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
                },
                "gap_type": {
                    "type": "choice",
                    "instructions": "What kind of gap, if any, exists between the reasoning and the answer?",
                    "criteria": {
                        "none": "Reasoning fully supports the answer, no gap",
                        "incomplete_coverage": "Reasoning doesn't address all relevant cases/branches/parts of the problem",
                        "unverified_claim": "Reasoning asserts something as fact without checking it",
                        "contradiction": "Reasoning contradicts itself or the stated answer",
                        "off_topic": "Reasoning doesn't actually address the question asked"
                    }
                }
            }
        },
        timeout=15,
    )
    resp.raise_for_status()
    answers = resp.json()["answers"]
    return answers["well_supported"]["noul"], answers["gap_type"]["choice"]


def jev_gate(question: str, answer: str, reasoning: str) -> tuple[bool, str]:
    """Jev as the agent's critic: pass/fail plus the gap it found to retry on."""
    confidence, gap_type = judge_reasoning(question, answer, reasoning)
    print(f"JEV: confidence={confidence:.2f} gap_type={gap_type}")
    return gap_type == "none", gap_type

if __name__ == "__main__":
    with open("questions.json") as f:
        questions = json.load(f)

        item = questions[QUESTION_INDEX]
        print("TASK_ID:", item["task_id"])
        print("QUESTION:", item["question"])

        # file_path = download_file(item["task_id"], item["file_name"]) if item.get("file_name") else None
        file_path = get_file_path(item["task_id"]) if item.get("file_name") else None
        answer, reasoning, messages = agent_answer(item["question"], file_path, evaluator=jev_gate)

        print('/n------------- TRANSCRIPT -------------/n')
        for m in messages:
            print(json.dumps(m, indent=2, default=str))
        print('/n--------------------------------/n')
        print("ANSWER:", answer)
        print("REASONING:", reasoning)

        if answer is None:
            print(f"No answer survived Jev within the step limit — not submitting. task_id={item['task_id']}")
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