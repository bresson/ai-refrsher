"""Generic, provider-agnostic tool-calling agent loop.

This file never changes between problems or between LLM providers.
Provider selection happens entirely via the `model` string passed in
(e.g. "gpt-4o", "anthropic/claude-sonnet-4-5", "gemini/gemini-2.0-flash").
"""

import json
from litellm import completion
import re
from tools import VISION_TOOL_FUNCTIONS, TOOL_FUNCTIONS, TOOLS, VIDEO_TOOL_FUNCTIONS, VIDEO_TOOLS, VISION_TOOLS

YOUTUBE_URL_RE = re.compile(r"youtube\.com/watch|youtu\.be/")

AGENT_CONFIGS = {
    "default": {
        "model": "gpt-5.6-sol",
        "tools": TOOLS,
        "tool_functions": TOOL_FUNCTIONS,
    },
    "image": {
        "model": "anthropic/claude-opus-5",
        "tools": VISION_TOOLS,
        "tool_functions": VISION_TOOL_FUNCTIONS,
    },
    "video": {
        "model": "anthropic/claude-opus-5",
        "tools": VIDEO_TOOLS,
        "tool_functions": VIDEO_TOOL_FUNCTIONS,
    },
}

def classify(question: str, file_path: str = '') -> str:
    if YOUTUBE_URL_RE.search(question):
        return "video"
    elif file_path:
        return 'image'
    return "default"

VERDICT_PREFIX = "[automated verification]"


def _work_behind(messages) -> str:
    """The work standing behind the answer just given: the agent's own narration
    and the tool output it read, since its last attempt was rejected.

    This is what gets judged. A separate reasoning field only comes back from
    the provider when thinking is enabled, so the transcript is the reliable
    source — and it is the actual record of how the answer was reached.
    """
    chunk = []
    for m in reversed(messages):
        if m["role"] == "user" and str(m.get("content", "")).startswith(VERDICT_PREFIX):
            break
        if m.get("reasoning_content"):
            chunk.append(m["reasoning_content"])
        if m["role"] == "assistant" and m.get("content"):
            chunk.append(m["content"])
        elif m["role"] == "tool":
            chunk.append(f"tool result: {m['content']}")
    return "\n\n".join(reversed(chunk))


def _verdict(passed: bool, candidate: str, feedback) -> str:
    """Feedback the agent can act on, and that it won't mistake for the user."""
    if passed:
        return f"{VERDICT_PREFIX} answer accepted."
    return (
        f"{VERDICT_PREFIX} Your answer was checked against your own working and "
        f"was not accepted.\nAnswer checked: {candidate[:200]}\nGap found: {feedback}\n"
        "The user did not write this and has nothing to add — do not ask them "
        "questions. Close the gap with your tools if you need better information, "
        "then state your answer again."
    )


def run_agent(model, user_message, tools, tool_functions, system_prompt="", max_steps=5, evaluator=None):
    """Run a tool-calling agent loop until it produces a final answer.

    Args:
        model: LiteLLM model string, e.g. "gpt-4o" or "anthropic/claude-sonnet-4-5".
        user_message: The task/question for the agent to work on.
        tools: List of tool schemas in OpenAI function-calling format.
        tool_functions: Dict mapping tool name -> callable.
        system_prompt: Optional system instructions.
        max_steps: Max tool-call rounds before giving up.
        evaluator: Optional callable (question, answer, reasoning) -> (passed, feedback).
            Judged only on a turn where the model stops calling tools: that
            turn's text is the answer, and the work behind it is the reasoning.
            A rejection goes back into the transcript and the loop keeps working.

    Returns:
        (answer, reasoning, messages). answer is None if no candidate was ever
        accepted. messages is the full transcript: every step's reasoning and
        every verdict, in order.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    answer, reasoning = None, None

    for _ in range(max_steps):
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            # tool_choice={"type": "function", "function": {"name": tools[0]["function"]["name"]}},
            thinking={"type": "adaptive", "display": "summarized"},
            # output_config={"effort": "high"},
        )
        # print("REASONING_CONTENT:", repr(response.choices[0].message.reasoning_content))
        # print("THINKING_BLOCKS:", repr(response.choices[0].message.thinking_blocks))
        msg = response.choices[0].message
        # print("FULL_MESSAGE:", msg.model_dump())

        # Rebuild the assistant turn as a plain dict rather than appending
        # the raw response object. LiteLLM routes to many different backends
        # (OpenAI, Anthropic, Gemini, ...) and a plain dict is the one shape
        # guaranteed to serialize correctly no matter which provider is live.
        if msg.content is None:
            content = ""
        elif isinstance(msg.content, str):
            content = msg.content
        else:
            content = json.dumps(msg.content)
        assistant_turn = {"role": "assistant", "content": content}

        # Kept in the transcript when the provider returns it. It is not relied
        # on: most providers return nothing here unless thinking is enabled.
        step_reasoning = getattr(msg, "reasoning_content", None)
        if step_reasoning:
            assistant_turn["reasoning_content"] = step_reasoning

        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)

        # Still working: run the tools and go round again. Text on a turn that
        # also calls a tool is narration, not an answer, so nothing is judged.
        if msg.tool_calls:
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                fn = tool_functions.get(name)
                result = fn(**args) if fn else f"Unknown tool: {name}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })
            continue

        # No tool call: the model is done, so this turn's text is the answer.
        candidate = content.strip()
        if not candidate:
            break

        work = _work_behind(messages)
        if evaluator is None:
            answer, reasoning = candidate, work
            break

        passed, feedback = evaluator(user_message, candidate, work)
        messages.append({"role": "user", "content": _verdict(passed, candidate, feedback)})
        if passed:
            answer, reasoning = candidate, work
            break

    return answer, reasoning, messages

def agent_answer(question_text: str, file_path: str | None = None, evaluator=None):
    config = AGENT_CONFIGS[classify(question_text, file_path)]
    user_message = f"{question_text}\n\nImage file path: {file_path}" if file_path else question_text
    return run_agent(
        model=config["model"],
        user_message=user_message,
        tools=config["tools"],
        tool_functions=config["tool_functions"],
        evaluator=evaluator,
    )

class Agent:
    def __call__(self, question, **kwargs):
        return run_agent(MODEL, question, TOOLS, TOOL_FUNCTIONS, **kwargs)