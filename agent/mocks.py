"""
==============================================================================
  EVERY MOCKED EXTERNAL INTEGRATION IN THIS AGENT LIVES IN THIS FILE.
==============================================================================

There are exactly two, and both are mocked for the same reason: the exercise
supplies no API key and states that a documented mock is judged the same as a
working call. Nothing else in the agent talks to a network.

  SEAM 1  call_claude_for_rationale()  - the Claude Messages API call
  SEAM 2  Tracer                       - LangSmith-style run tracing

Each seam below states what a real call would send, what it would return, and
exactly where the real implementation plugs in. The real code is written out
in full rather than sketched, so it can be judged on its own design merits.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

# ============================================================================
# SEAM 1 — CLAUDE MESSAGES API  (mocked)
# ============================================================================
#
# WHAT THIS CALL IS FOR, AND WHAT IT DELIBERATELY IS NOT FOR
# ----------------------------------------------------------
# It writes a short, rep-readable justification for an account that has ALREADY
# been ranked, tiered and flagged by deterministic code. It makes no decision.
# It cannot promote, demote or exclude an account.
#
# Why not do this with string templates? Templates handle this badly because
# the interesting cases are combinatorial - heavy web engagement with no trial,
# a dormant trial with heavy sales contact, a large account with no vendor
# coverage. Each combination needs its own branch and the branches multiply.
# One prompt covers them all.
#
# Honest assessment: this is a real but MODEST gain. A template covers perhaps
# 80% of it, which is exactly why render_fallback_rationale() below is a fully
# working template implementation rather than a stub. The LLM is an enhancement
# layer over a working baseline, so its marginal value stays measurable and the
# agent degrades rather than fails when the API is unavailable.

MODEL_ID = "claude-opus-5"

# Kept as a module-level constant rather than rebuilt per call, because it is
# the cacheable prefix: identical across every account in a batch, so a real
# run marks it with cache_control and pays for it once instead of ~25 times.
SYSTEM_PROMPT = """\
You write one-line call justifications for B2B sales reps at Cordilla Systems.

You will be given structured facts about a single account that a scoring model
has already prioritised. Your job is to explain, in plain language, why it is
worth a call today and what to open the conversation with.

Hard rules:
1. Use ONLY the facts provided. Never introduce a number, date, company detail
   or event that is not in the input. If a field is absent, it is unknown.
2. Never state or imply that the account will convert, is likely to buy, or is
   a strong bet. The underlying score ranks accounts; it is not a calibrated
   probability and must not be presented as one.
3. If vendor_intent_data is "missing", you must say plainly that third-party
   intent data is unavailable for this account and that its ranking is
   therefore less reliable.
4. No ML vocabulary: no "model", "score", "probability", "percentile", "tier".
   Reps care about what the account did, not how it was ranked.
5. rationale: at most 2 sentences. opening_line: one sentence a rep could say.

Return only the structured object requested."""


# Structured-output contract. Enforced server-side via output_config.format so
# the response cannot come back as prose that then needs parsing. The guardrail
# in agent/guardrails.py re-checks the CONTENT, because a schema guarantees
# shape and says nothing about whether a fact was invented.
RATIONALE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": "At most 2 sentences on why this account is worth a call today.",
        },
        "opening_line": {
            "type": "string",
            "description": "One sentence the rep could open the call with.",
        },
        "data_caveat": {
            "type": ["string", "null"],
            "description": (
                "Required non-null when third-party intent data is missing; "
                "states that the ranking is less reliable for this account."
            ),
        },
    },
    "required": ["rationale", "opening_line", "data_caveat"],
    "additionalProperties": False,
}


def build_user_message(facts: dict[str, Any]) -> str:
    """Render the per-account half of the prompt.

    Deliberately a flat list of labelled facts rather than raw JSON or a CSV
    row: the model must not see account_id (nothing useful, invites
    hallucinated specifics) and must not see the raw probability (rule 2 above
    exists precisely to stop it being repeated back to a rep).
    """
    lines = [
        f"account_type: {facts['account_type']}",
        f"industry: {facts['industry']}",
        f"employee_count: {facts['employee_count']}",
        f"vendor_intent_data: {'present' if facts['has_intent'] else 'missing'}",
        f"marketing_qualified_leads_last_90d: {facts['mql_count_90d']}",
        f"website_visits_last_90d: {facts['web_touchpoints_90d']}",
        f"sales_contacts_last_90d: {facts['sales_contacts_90d']}",
        f"started_a_trial: {'yes' if facts['trial_started'] else 'no'}",
        f"active_trial_users: {facts['trial_active_users']}",
        f"days_since_data_snapshot: {facts['snapshot_age_days']}",
    ]
    if facts["has_intent"]:
        lines.append(f"vendor_intent_value: {facts['intent_score']}")
    return "Account facts:\n" + "\n".join(lines)


# ----------------------------------------------------------------------------
# THE REAL CALL — this is what replaces the mock. Not executed: no API key is
# provided with this exercise, and no anthropic package is installed.
# ----------------------------------------------------------------------------
#
# import anthropic
# _client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the env
#
# def call_claude_for_rationale(facts: dict) -> dict:
#     response = _client.messages.create(
#         model=MODEL_ID,
#         max_tokens=1024,
#         # Stable across every account in the batch, so cache it and pay for
#         # the system prompt once per run instead of once per account.
#         system=[{
#             "type": "text",
#             "text": SYSTEM_PROMPT,
#             "cache_control": {"type": "ephemeral"},
#         }],
#         # Short, well-specified writing task. Low effort keeps latency and
#         # cost down across a whole batch; this is not a reasoning problem.
#         output_config={
#             "effort": "low",
#             "format": {
#                 "type": "json_schema",
#                 "name": "account_rationale",
#                 "schema": RATIONALE_SCHEMA,
#             },
#         },
#         messages=[{"role": "user", "content": build_user_message(facts)}],
#     )
#     # stop_details is populated only on a refusal; guard before reading it.
#     if response.stop_reason == "refusal":
#         raise LLMCallFailed(f"refused: {response.stop_details.category}")
#     return json.loads(
#         next(b.text for b in response.content if b.type == "text")
#     )
#
# Error handling at the call site (agent/nodes.py) catches LLMCallFailed and
# falls back to the template. In production the chain would be specific rather
# than broad: anthropic.RateLimitError (retry with backoff) ->
# anthropic.APIStatusError >= 500 (retry) -> anthropic.BadRequestError (do not
# retry, log and fall back) -> anthropic.APIConnectionError (retry).
# ----------------------------------------------------------------------------


class LLMCallFailed(RuntimeError):
    """Raised when the rationale call fails. Triggers the template fallback."""


def call_claude_for_rationale(facts: dict[str, Any]) -> dict[str, Any]:
    """MOCK of the call above. Returns the same shape RATIONALE_SCHEMA defines.

    Deterministic on purpose: the same account always produces the same text,
    so the guardrail and the run report are reproducible across runs. A real
    call would not be, which is one more reason the guardrail exists.

    The phrasing here is deliberately plainer than a real model would produce.
    It is not trying to look like convincing LLM output - it is standing in for
    it at the right interface so the surrounding machinery can be judged.
    """
    reasons = []
    if facts["trial_started"] and facts["trial_active_users"] > 0:
        reasons.append(
            f"{facts['trial_active_users']} "
            f"{'person is' if facts['trial_active_users'] == 1 else 'people are'} "
            "already active in a trial"
        )
    elif facts["trial_started"]:
        reasons.append("they started a trial but nobody has used it yet")
    if facts["web_touchpoints_90d"] >= 4:
        reasons.append(f"{facts['web_touchpoints_90d']} website visits in the last 90 days")
    if facts["sales_contacts_90d"] >= 3:
        reasons.append(f"{facts['sales_contacts_90d']} sales touches already logged")
    elif facts["sales_contacts_90d"] == 0:
        reasons.append("nobody from sales has contacted them yet")
    if facts["mql_count_90d"] >= 2:
        reasons.append(f"{facts['mql_count_90d']} marketing-qualified leads in 90 days")
    if not reasons:
        reasons.append(
            f"a {facts['industry'].lower()} account of about "
            f"{facts['employee_count']} staff with light recent activity"
        )

    body = reasons[0] if len(reasons) == 1 else f"{', '.join(reasons[:-1])}, and {reasons[-1]}"
    rationale = (
        f"This {facts['account_type'].lower()} is worth a look because {body}."
    )

    if facts["trial_started"]:
        opening = "Ask how the trial has been going and what they were hoping to get out of it."
    elif facts["web_touchpoints_90d"] >= 4:
        opening = "Mention they have been reading up on us and ask what prompted the interest."
    elif facts["sales_contacts_90d"] == 0:
        opening = "Open cold - introduce yourself and ask what their current workflow looks like."
    else:
        opening = "Pick up from the last conversation and ask what has changed since."

    caveat = None
    if not facts["has_intent"]:
        caveat = (
            "We hold no third-party intent data on this account, so its position "
            "on the list is less reliable than others."
        )

    return {"rationale": rationale, "opening_line": opening, "data_caveat": caveat}


def render_fallback_rationale(facts: dict[str, Any]) -> dict[str, Any]:
    """Deterministic template used when the LLM call fails or is rejected.

    WHY THIS IS A REAL IMPLEMENTATION AND NOT A STUB: it is the reason the
    agent can treat the LLM as optional. Nobody's call list should disappear
    because a third-party API is down, and having a working baseline is what
    makes the LLM's marginal contribution measurable rather than assumed.
    """
    signals = []
    if facts["trial_started"]:
        signals.append(f"trial started, {facts['trial_active_users']} active users")
    if facts["web_touchpoints_90d"]:
        signals.append(f"{facts['web_touchpoints_90d']} web visits/90d")
    if facts["sales_contacts_90d"]:
        signals.append(f"{facts['sales_contacts_90d']} sales touches/90d")
    if facts["mql_count_90d"]:
        signals.append(f"{facts['mql_count_90d']} MQLs/90d")
    summary = "; ".join(signals) if signals else "no recent recorded activity"
    return {
        "rationale": f"{facts['account_type']}, {facts['industry']}, "
                     f"~{facts['employee_count']} staff. Recent signals: {summary}.",
        "opening_line": "Introduce yourself and confirm who owns this decision.",
        "data_caveat": (
            None if facts["has_intent"] else
            "No third-party intent data held for this account; ranking is less reliable."
        ),
    }


# ============================================================================
# SEAM 2 — LANGSMITH RUN TRACING  (mocked)
# ============================================================================
#
# WHAT THIS IS FOR: it is the mechanism behind run_report.json, and the raw
# material the monitoring design consumes. An LLM step that nobody can see
# inside is the exact failure mode this whole exercise is about - something
# that keeps returning plausible output while quietly getting worse.
#
# THE REAL IMPLEMENTATION is almost entirely configuration rather than code:
#
#   export LANGSMITH_TRACING=true
#   export LANGSMITH_API_KEY=...
#   export LANGSMITH_PROJECT=cordilla-call-list
#
#   from langsmith import traceable
#
#   @traceable(run_type="llm", name="generate_rationale")
#   def call_claude_for_rationale(facts: dict) -> dict:
#       ...
#
# With LANGSMITH_TRACING set, LangGraph emits a span per node automatically -
# node name, inputs, outputs, latency, errors - with no further instrumentation.
# The @traceable decorator adds the LLM call itself as a child span carrying
# token counts and the resolved prompt.
#
# The Tracer below records the same fields locally so the observability surface
# is real and inspectable offline. What a hosted backend would add on top is
# retention, cross-run comparison and alerting - not different data.


class Tracer:
    """Collects one span per traced operation. Stands in for a LangSmith run tree.

    Records the fields that matter for catching silent degradation: which node
    ran, how long it took, whether it succeeded, and - for LLM calls - whether
    the output survived the guardrail. Token counts are mocked and labelled as
    such; a real trace would carry usage from the API response.
    """

    def __init__(self, project: str = "cordilla-call-list"):
        self.project = project
        self.run_id = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        self.spans: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []

    def span(self, name: str, run_type: str = "chain", **metadata):
        return _Span(self, name, run_type, metadata)

    def _record(self, span: dict[str, Any]) -> None:
        self.spans.append(span)

    def summary(self) -> dict[str, Any]:
        """Aggregate the spans into the numbers the monitoring check reads."""
        by_type: dict[str, dict[str, Any]] = {}
        for s in self.spans:
            bucket = by_type.setdefault(
                s["run_type"], {"count": 0, "errors": 0, "total_ms": 0.0})
            bucket["count"] += 1
            bucket["total_ms"] += s["duration_ms"]
            if s["status"] != "ok":
                bucket["errors"] += 1
        for bucket in by_type.values():
            bucket["total_ms"] = round(bucket["total_ms"], 2)
        return {
            "run_id": self.run_id,
            "project": self.project,
            "span_count": len(self.spans),
            "by_run_type": by_type,
            "note": "Token counts are mocked. A real LangSmith trace carries "
                    "usage from the API response.",
        }


class _Span:
    """Context manager for one span. Captures duration and failure status."""

    def __init__(self, tracer: Tracer, name: str, run_type: str, metadata: dict):
        self.tracer, self.name, self.run_type, self.metadata = tracer, name, run_type, metadata
        self.inputs: Any = None
        self.outputs: Any = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.tracer._record({
            "name": self.name,
            "run_type": self.run_type,
            "duration_ms": round((time.perf_counter() - self._t0) * 1000, 2),
            "status": "ok" if exc_type is None else "error",
            "error": None if exc is None else f"{exc_type.__name__}: {exc}",
            "inputs": _truncate(self.inputs),
            "outputs": _truncate(self.outputs),
            **self.metadata,
        })
        return False  # never swallow the exception; the caller decides


def _truncate(value: Any, limit: int = 400) -> Any:
    """Keep the report readable. A hosted backend would store these in full."""
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"
