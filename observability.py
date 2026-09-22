"""Arize Phoenix wiring for the agent.

This is the only file that imports Phoenix or OpenTelemetry. agent.py,
tools.py, and evaluation.py stay observability-agnostic — call
setup_tracing() once from whichever script is the entry point
(main.py for a smoke test, or your batch-run script for a full pass),
before the agent runs.
"""

from phoenix.otel import register
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry import trace

_tracer_provider = None


def setup_tracing(project_name: str = "gaia-agent", endpoint: str = "http://localhost:6006/v1/traces"):
    """Start a Phoenix tracer and instrument LiteLLM calls.

    Safe to call more than once — later calls are no-ops. Requires a
    Phoenix collector running (e.g. `phoenix serve` locally, or
    Phoenix Cloud) at the given endpoint.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider

    _tracer_provider = register(project_name=project_name, auto_instrument=True, endpoint=endpoint, batch=True)
    LiteLLMInstrumentor().instrument(tracer_provider=_tracer_provider)
    return _tracer_provider


# def traced_tools(tool_functions: dict) -> dict:
#     """Wrap each tool function in its own span.

#     LiteLLM instrumentation only sees the LLM call, not local tool
#     execution. This wraps TOOL_FUNCTIONS from the outside, so tools.py
#     itself never needs to know Phoenix exists.
#     """
#     tracer = trace.get_tracer(__name__)

#     def _wrap(name, fn):
#         def wrapped(**kwargs):
#             with tracer.start_as_current_span(f"tool.{name}") as span:
#                 span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
#                 span.set_attribute("tool.name", name)
#                 span.set_attribute("tool.input", str(kwargs))
#                 result = fn(**kwargs)
#                 span.set_attribute("tool.output", str(result))
#                 return result
#         return wrapped

#     return {name: _wrap(name, fn) for name, fn in tool_functions.items()}