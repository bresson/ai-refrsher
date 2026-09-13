"""Offline unit test for the agent loop.

Validates the control flow (tool dispatch, message construction,
final-answer termination, step-limit handling) WITHOUT calling any
real LLM API or spending any API credits. Run this FIRST, before
touching main.py, to confirm the plumbing is correct.

    python test_agent.py
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent import run_agent, Agent


def _tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(tool_calls=None, content=None):
    message = SimpleNamespace(tool_calls=tool_calls, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_calls_tool_then_returns_final_answer():
    """Round 1: model calls web_search. Round 2: model calls final_answer.
    Confirms run_agent dispatches the tool, feeds the result back in,
    and returns the final answer -- no real API key needed."""

    responses = [
        _response(tool_calls=[_tool_call("call_1", "web_search", {"query": "test"})]),
        _response(tool_calls=[_tool_call("call_2", "final_answer", {"answer": "42"})]),
    ]

    def fake_web_search(query):
        assert query == "test"
        return "mocked search result"

    tool_functions = {"web_search": fake_web_search}

    with patch("agent.completion", side_effect=responses) as mock_completion:
        result = run_agent("fake-model", "irrelevant question", [], tool_functions)

    assert result == "42", f"Expected '42', got {result!r}"
    assert mock_completion.call_count == 2, "Expected exactly 2 LLM calls"
    print("PASS: test_agent_calls_tool_then_returns_final_answer")


def test_agent_returns_plain_text_with_no_tool_calls():
    """Model answers directly with no tool use at all."""

    responses = [_response(tool_calls=None, content="Paris")]

    with patch("agent.completion", side_effect=responses):
        result = run_agent("fake-model", "What is the capital of France?", [], {})

    assert result == "Paris", f"Expected 'Paris', got {result!r}"
    print("PASS: test_agent_returns_plain_text_with_no_tool_calls")


def test_agent_hits_step_limit():
    """Model keeps calling a tool forever and never finishes."""

    responses = [
        _response(tool_calls=[_tool_call(f"call_{i}", "web_search", {"query": "loop"})])
        for i in range(10)
    ]

    with patch("agent.completion", side_effect=responses):
        result = run_agent(
            "fake-model", "question", [], {"web_search": lambda query: "result"},
            max_steps=10,
        )

    assert result == "No answer produced within step limit."
    print("PASS: test_agent_hits_step_limit")


if __name__ == "__main__":
    test_agent_calls_tool_then_returns_final_answer()
    test_agent_returns_plain_text_with_no_tool_calls()
    test_agent_hits_step_limit()
    print("\nAll tests passed.")