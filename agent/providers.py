"""
Pluggable LLM backends for the rationale step.

    python -m agent.graph                          # mock, the default
    python -m agent.graph --llm groq
    python -m agent.graph --llm groq --model openai/gpt-oss-120b
    python -m agent.graph --list-models groq
    python -m agent.graph --compare mock,groq      # same accounts, both models

WHY THIS EXISTS
---------------
The exercise supplies no API key and says a documented mock is judged the same
as a live call, so `mock` stays the default and the repo runs end to end with
nothing installed beyond the pinned core plus langgraph.

The reason to build the switch anyway is that it turns the guardrail into a
model-evaluation harness. agent/guardrails.py does not know or care which model
produced a rationale - it checks the text against the facts that went into the
prompt. So swapping the backend and re-running scores every model on identical
inputs against identical criteria: invented numbers, likelihood language,
leaked internal vocabulary, dropped vendor-data caveats. That is a measurable
comparison rather than a read-through, and the mechanism is the same whether
the backend is Claude, a Groq-hosted open model, or the deterministic stand-in.

TWO DESIGN RULES, both of which matter more than they look
----------------------------------------------------------
1. A live provider that cannot run FAILS LOUDLY at startup. It never silently
   falls back to the mock. Presenting mock output as though a real model
   produced it is precisely the class of quiet wrongness this project is about.
2. SDKs are imported lazily, inside the provider that needs them. Installing
   `groq` must never become a condition of running the default path.

The per-account fallback in agent/nodes.py is a different mechanism and still
applies: if a CONFIGURED provider fails on one account, that account gets the
template rationale and the run report counts it.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any

from agent import mocks
from agent.mocks import LLMCallFailed, RATIONALE_SCHEMA, SYSTEM_PROMPT, build_user_message


class ProviderNotConfigured(RuntimeError):
    """Raised before any work starts: missing key, missing SDK, unknown name."""


def _lazy_import(module_name: str, pip_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ProviderNotConfigured(
            f"the '{module_name}' package is not installed. Run: pip install {pip_name}"
        ) from exc


def _require_key(env_var: str, provider: str) -> None:
    if not os.environ.get(env_var):
        raise ProviderNotConfigured(
            f"{env_var} is not set, so --llm {provider} cannot run. "
            f"Set it, or use --llm mock (the default)."
        )


class Provider:
    """Common interface. Subclasses implement generate()."""

    name: str = "base"
    default_model: str = ""
    requires_key: str | None = None
    is_live: bool = True

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    def generate(self, facts: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}:{self.model}" + ("" if self.is_live else " (mocked)")

    @classmethod
    def list_models(cls) -> list[str]:
        raise ProviderNotConfigured(f"{cls.name} cannot enumerate models")


class MockProvider(Provider):
    """The deterministic stand-in. Default, and the only keyless backend.

    Delegates to agent/mocks.py so the documented seam stays in one place.
    """

    name = "mock"
    default_model = "deterministic-template"
    is_live = False

    def generate(self, facts):
        return mocks.call_claude_for_rationale(facts)


class AnthropicProvider(Provider):
    """Claude via the Messages API, with server-enforced structured output.

    The system prompt is marked cacheable because it is byte-identical across
    every account in a batch: a 25-account run pays for it once instead of 25
    times. Effort is low - this is a short, well-specified writing task, not a
    reasoning problem.

    NOT TESTED: no ANTHROPIC_API_KEY was available while building this. Written
    against the documented API shape and labelled unverified rather than
    presented as working.
    """

    name = "anthropic"
    default_model = "claude-opus-5"
    requires_key = "ANTHROPIC_API_KEY"

    def __init__(self, model=None):
        super().__init__(model)
        _require_key(self.requires_key, self.name)
        anthropic = _lazy_import("anthropic", "anthropic")
        self._client = anthropic.Anthropic()

    def generate(self, facts):
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema",
                               "name": "account_rationale",
                               "schema": RATIONALE_SCHEMA},
                },
                messages=[{"role": "user", "content": build_user_message(facts)}],
            )
        except Exception as exc:
            raise LLMCallFailed(f"anthropic call failed: {exc}") from exc
        # stop_details is populated only on a refusal; guard before reading it.
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMCallFailed(f"refused: {response.stop_details.category}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return _parse_json(text, self.name)


class GroqProvider(Provider):
    """Open models hosted on Groq, via its OpenAI-compatible completions API.

    Uses response_format json_object rather than a JSON schema, because schema
    support varies across the models Groq serves and the entire point of this
    backend is to run ACROSS models. The required shape is therefore stated in
    the prompt and enforced afterwards by agent/guardrails.py.

    That is a weaker guarantee than Anthropic's server-side schema, and it is
    deliberately left visible: comparing a provider with enforced structure
    against one without is part of what the guardrail is there to measure.
    """

    name = "groq"
    # Verified present via `--list-models groq` on 2026-09-20. Groq rotates its
    # catalogue: the first default written here (llama-3.3-70b-versatile) had
    # already been retired, which is why --list-models exists rather than a
    # hardcoded list. If this model disappears too, --list-models then --model.
    default_model = "openai/gpt-oss-120b"
    requires_key = "GROQ_API_KEY"

    def __init__(self, model=None):
        super().__init__(model)
        _require_key(self.requires_key, self.name)
        groq = _lazy_import("groq", "groq")
        self._client = groq.Groq()

    def _system_prompt(self) -> str:
        return (SYSTEM_PROMPT
                + "\n\nReturn one JSON object with exactly these keys:\n"
                + json.dumps(RATIONALE_SCHEMA["properties"], indent=2)
                + "\n\nNo prose outside the JSON object.")

    def generate(self, facts):
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": build_user_message(facts)},
                ],
            )
        except Exception as exc:
            raise LLMCallFailed(f"groq call failed: {exc}") from exc
        return _parse_json(response.choices[0].message.content, self.name)

    @classmethod
    def list_models(cls):
        _require_key(cls.requires_key, cls.name)
        groq = _lazy_import("groq", "groq")
        return sorted(m.id for m in groq.Groq().models.list().data)


class OpenAIProvider(Provider):
    """GPT models via Chat Completions with a strict JSON schema.

    NOT TESTED: no OPENAI_API_KEY was available while building this. Same
    caveat as AnthropicProvider - written to the documented shape, labelled
    rather than assumed.
    """

    name = "openai"
    default_model = "gpt-4o-mini"
    requires_key = "OPENAI_API_KEY"

    def __init__(self, model=None):
        super().__init__(model)
        _require_key(self.requires_key, self.name)
        openai = _lazy_import("openai", "openai")
        self._client = openai.OpenAI()

    def generate(self, facts):
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                max_tokens=1024,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "account_rationale",
                                    "schema": RATIONALE_SCHEMA,
                                    "strict": True},
                },
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(facts)},
                ],
            )
        except Exception as exc:
            raise LLMCallFailed(f"openai call failed: {exc}") from exc
        return _parse_json(response.choices[0].message.content, self.name)

    @classmethod
    def list_models(cls):
        _require_key(cls.requires_key, cls.name)
        openai = _lazy_import("openai", "openai")
        return sorted(m.id for m in openai.OpenAI().models.list().data)


REGISTRY: dict[str, type[Provider]] = {
    "mock": MockProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "openai": OpenAIProvider,
}


def _parse_json(text: str, provider: str) -> dict[str, Any]:
    """Parse a model's JSON reply.

    Tolerates a fenced code block, which smaller models add despite being told
    not to. Tolerates nothing else: a provider that cannot return parseable
    JSON should surface as a failure the run report counts, not be repaired
    into looking compliant.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMCallFailed(f"{provider} returned unparseable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMCallFailed(f"{provider} returned {type(payload).__name__}, expected object")
    # Absent means "no caveat needed". The guardrail decides whether that is
    # acceptable for this account; this only normalises the shape.
    payload.setdefault("data_caveat", None)
    return payload


def get_provider(name: str, model: str | None = None) -> Provider:
    """Resolve a provider by name, raising before any work starts if unusable."""
    if name not in REGISTRY:
        raise ProviderNotConfigured(
            f"unknown provider {name!r}. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name](model)
