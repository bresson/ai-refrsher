import os
import json
import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
CACHE_PATH = "answers_cache.json"


def fetch_questions(api_url: str = DEFAULT_API_URL) -> list[dict]:
    resp = requests.get(f"{api_url}/questions", timeout=15)
    resp.raise_for_status()
    return resp.json()


def download_file(task_id: str, file_name: str, api_url: str = DEFAULT_API_URL) -> str:
    resp = requests.get(f"{api_url}/files/{task_id}", timeout=30)
    resp.raise_for_status()
    file_path = f"/tmp/{file_name}"
    with open(file_path, "wb") as f:
        f.write(resp.content)
    return file_path

"""
One thing not yet wired up: whatever script eventually calls 
run_agent_on_questions for the full batch also needs to call 
setup_tracing() and wrap its TOOL_FUNCTIONS the same way 
main.py does — that script isn't among the three files you 
shared, so I didn't touch it.
"""

def run_agent_on_questions(
    agent,
    questions_data: list[dict],
    submit_ready: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run the agent over a batch of GAIA questions.

    submit_ready controls caching:
      - False (default): every invocation is treated as unique. The
        cache is neither read nor written, so each run is a fresh,
        fully-traced call to the agent. Use this while iterating.
      - True: the original cache-backed behavior. Answers are read
        from / written to CACHE_PATH, so a real submission run is
        reproducible and doesn't re-spend API calls on retry.
    """
    cache = json.load(open(CACHE_PATH)) if (submit_ready and os.path.exists(CACHE_PATH)) else {}
    results_log, answers_payload = [], []

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name", "")
        if not task_id or question_text is None:
            continue

        if submit_ready and task_id in cache:
            submitted_answer = cache[task_id]
        else:
            file_path = download_file(task_id, file_name) if file_name else None
            submitted_answer = agent(question_text, file_path).strip()
            if submit_ready:
                cache[task_id] = submitted_answer
                json.dump(cache, open(CACHE_PATH, "w"))

        results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})

    return results_log, answers_payload


def submit_answers(username: str, agent_code: str, answers_payload: list[dict], api_url: str = DEFAULT_API_URL) -> dict:
    resp = requests.post(f"{api_url}/submit", json={
        "username": "bresson", "agent_code": agent_code, "answers": answers_payload
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()