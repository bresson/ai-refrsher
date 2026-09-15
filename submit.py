from evaluation import submit_answers

answers_payload = [
    {"task_id": "8e867cd7-cff9-4e6c-867a-ff5ddc2550be", "submitted_answer": "3"},
]

result = submit_answers(
    username="bresson",
    agent_code="https://github.com/bresson/ai-refrsher.git",
    answers_payload=answers_payload,
)
print(result)