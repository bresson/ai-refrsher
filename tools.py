"""Tool implementations and schemas for this problem.

This is the only file you edit when moving to a new problem
(GAIA, Aviary, whatever comes next) — agent.py stays untouched.
"""
import logging
import os
from opentelemetry import trace
import base64

from tavily import TavilyClient
from litellm import completion as litellm_completion

client = TavilyClient()  # reads TAVILY_API_KEY from env automatically

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("web_search")

MAX_CALLS = int(os.environ.get("TAVILY_MAX_CALLS", 900))  # stay under 1,000/month free tier
_call_count = 0

tracer = trace.get_tracer("web_search")
video_tracer = trace.get_tracer("analyze_video")

VISION_MODEL = "anthropic/claude-haiku-4-5"
vision_tracer = trace.get_tracer("analyze_image")


def web_search(query: str) -> str:
    global _call_count

    with tracer.start_as_current_span("web_search") as span:
        span.set_attribute("input.value", query)

        if _call_count >= MAX_CALLS:
            span.set_attribute("output.value", "budget_exhausted")
            return "[SEARCH_FAILED] Call budget exhausted for this session."

        _call_count += 1

        try:
            result = client.search(query, max_results=3, timeout=15)
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("output.value", f"error: {e}")
            return f"[SEARCH_FAILED] {e}"

        hits = result.get("results", [])
        formatted = "\n".join(f"{r['title']}: {r['content']}" for r in hits) if hits else "[SEARCH_EMPTY] No results found."
        span.set_attribute("output.value", formatted)
        return formatted


def analyze_video(url: str, question: str) -> str:
    with video_tracer.start_as_current_span("analyze_video") as span:
        span.set_attribute("input.value", f"{url} | {question}")

        try:
            response = litellm_completion(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "file", "file": {"file_id": url}},
                        ],
                    }
                ],
            )
            result = response.choices[0].message.content
            span.set_attribute("output.value", result)
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("output.value", f"error: {e}")
            return f"[VIDEO_ANALYSIS_FAILED] {e}"

def analyze_image(file_path: str, question: str) -> str:
    with vision_tracer.start_as_current_span("analyze_image") as span:
        span.set_attribute("input.value", f"{file_path} | {question}")
        try:
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            response = litellm_completion(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
            )
            result = response.choices[0].message.content
            span.set_attribute("output.value", result)
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("output.value", f"error: {e}")
            return f"[IMAGE_ANALYSIS_FAILED] {e}"


FINAL_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Call this when you have the final answer to the task.",
        "parameters": {
            "type": "object",
            "properties": {
            "reasoning": {
                "type": "string",
                "description": "categorize the problem, eg calculation, analytical, quizzical, etc. State insights about the problem space used to arrive at the decision. Include any analysis or potential answers and solutions discarded. Explain your reasoning. use analytical methods such as deductive logic, inference, syllogism, etc ... that you used"            },
                "answer": {"type": "string", "description": "The exact final answer value only — no markdown, no explanation, no surrounding text."}
            },
            "required": ["reasoning", "answer"],
        },
    },
}

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
    FINAL_ANSWER_TOOL,
]

VIDEO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_video",
            "description": "Answer a visual question about a YouTube video's content (e.g. counting things shown on screen).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The YouTube video URL."},
                    "question": {"type": "string", "description": "The specific visual question to answer about the video."}
                },
                "required": ["url", "question"],
            },
        },
    },
    FINAL_ANSWER_TOOL,
]

VISION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Answer a visual question about a local image file (e.g. reading a chess position).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Local path to the image file."},
                    "question": {"type": "string", "description": "The specific visual question to answer about the image."}
                },
                "required": ["file_path", "question"],
            },
        },
    },
    FINAL_ANSWER_TOOL,
]

VISION_TOOL_FUNCTIONS = {"analyze_image": analyze_image}
TOOL_FUNCTIONS = {"web_search": web_search}
VIDEO_TOOL_FUNCTIONS = {"analyze_video": analyze_video}