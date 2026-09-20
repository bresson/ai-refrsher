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
        "model": "anthropic/claude-haiku-4-5",
        "tools": VISION_TOOLS,
        "tool_functions": VISION_TOOL_FUNCTIONS,
    },
    "video": {
        "model": "anthropic/claude-haiku-4-5",
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

def run_agent(model, user_message, tools, tool_functions, system_prompt="", max_steps=10):
    """Run a tool-calling agent loop until it produces a final answer.

    Args:
        model: LiteLLM model string, e.g. "gpt-4o" or "anthropic/claude-sonnet-4-5".
        user_message: The task/question for the agent to work on.
        tools: List of tool schemas in OpenAI function-calling format.
        tool_functions: Dict mapping tool name -> callable.
        system_prompt: Optional system instructions.
        max_steps: Max tool-call rounds before giving up.

    Returns:
        The agent's final answer as a string.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    for _ in range(max_steps):
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            # thinking={"type": "adaptive", "display": "summarized"},
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
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        assistant_turn = {"role": "assistant", "content": content}

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

        if not msg.tool_calls:
            return msg.content, None

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            if name == "final_answer":
                print("REASONING:", args["reasoning"])
                return args["answer"].strip(), args["reasoning"]

            fn = tool_functions.get(name)
            result = fn(**args) if fn else f"Unknown tool: {name}"
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return "No answer produced within step limit.", None

def agent_answer(question_text: str, file_path: str | None = None) -> str:
    config = AGENT_CONFIGS[classify(question_text, file_path)]
    user_message = f"{question_text}\n\nImage file path: {file_path}" if file_path else question_text
    return run_agent(
        model=config["model"],
        user_message=user_message,
        tools=config["tools"],
        tool_functions=config["tool_functions"],
    )

class Agent:
    def __call__(self, question, **kwargs):
        return run_agent(MODEL, question, TOOLS, TOOL_FUNCTIONS, **kwargs)