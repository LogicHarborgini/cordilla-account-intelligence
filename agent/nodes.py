"""
The graph's nodes. One function per step, plain and side-effect-light except
for the two that deliberately write files.

Each node's docstring states the decision or risk it addresses, not just its
mechanics. Read top to bottom, they are the argument for the design.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent import scoring
from agent.guardrails import validate_rationale
from agent.mocks import LLMCallFailed, Tracer, render_fallback_rationale
from agent.state import AgentState

# A single module-level tracer so every node writes into the same run tree,
# mirroring how a real LangSmith run groups spans under one root.
TRACER = Tracer()

REQUIRED_COLUMNS = set(scoring.FEATURES) | {"account_id", "snapshot_date"}

# Gate thresholds, expressed in standard errors rather than fixed percentage
# points. WHY: the noise in a coverage estimate depends on batch size. For 300
# accounts at ~61% coverage the standard error is about 2.8pp, so a fixed "warn
# at 5pp" rule would be hair-trigger on a 50-account batch and asleep on a
# 5,000-account one. Scaling with the batch keeps the meaning constant.
COVERAGE_WARN_SIGMA = 3.0    # ~0.3% chance under pure sampling noise
COVERAGE_HALT_SIGMA = 6.0    # not sampling noise; something structural changed

# Operational staleness. Snapshot age does NOT predict conversion (tested and
# refuted during the data work), so this is not a model-validity check - it is
# "reps would be calling on year-old information", which is a business problem
# regardless of what the model thinks.
STALE_MEDIAN_AGE_DAYS = 365


# ============================================================================
# 1. validate_input
# ============================================================================

def validate_input(state: AgentState) -> dict[str, Any]:
    """Refuse to produce a call list from input that cannot support one.

    THE RISK THIS ADDRESSES is the whole premise of this exercise: a scoring
    system that keeps emitting plausible output after its inputs have changed
    underneath it. A model handed shifted data does not crash - it returns
    numbers in the usual range, reps work the list, and credibility leaks away
    over a couple of quarters with nobody able to point at a failure.

    So the agent checks its input before it scores, and stops rather than
    emitting a list it cannot stand behind. This is the single most important
    node here, and the only one with a branch after it.
    """
    with TRACER.span("validate_input", run_type="chain") as span:
        frame = pd.read_csv(state["input_csv"])
        span.inputs = {"path": state["input_csv"], "rows": len(frame)}

        findings: list[dict[str, Any]] = []

        def add(severity: str, code: str, detail: str, **extra):
            findings.append({"severity": severity, "code": code, "detail": detail, **extra})

        # --- structural: the batch is unusable if any of these fire ---------
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            add("halt", "missing_columns", f"absent: {sorted(missing)}")
        if len(frame) == 0:
            add("halt", "empty_batch", "no rows to score")
        if "account_id" in frame.columns:
            dupes = int(frame.account_id.duplicated().sum())
            if dupes:
                add("halt", "duplicate_account_ids",
                    f"{dupes} duplicates; an account would appear twice on a rep's list")

        # Unseen categories do not raise: OneHotEncoder(handle_unknown='ignore')
        # silently encodes them as all-zeros, so a whole new industry would
        # score as though it had no industry at all. Exactly the quiet failure
        # this gate exists for.
        known = {
            "account_type": {"Prospect", "Suspect", "Former Customer"},
            "industry": {"Financial Services", "Healthcare", "Manufacturing",
                         "Professional Services", "Retail", "Software"},
        }
        for column, expected in known.items():
            if column in frame.columns:
                unseen = set(frame[column].dropna().unique()) - expected
                if unseen:
                    add("halt", "unseen_category",
                        f"{column} contains {sorted(unseen)}, which the encoder "
                        f"silently zeroes rather than rejecting")

        for column in scoring.FEATURES:
            if column in frame.columns and column != "intent_score" and frame[column].isna().all():
                add("halt", "feature_entirely_null", column)

        # --- statistical: vendor coverage -----------------------------------
        # The model's most important feature is intent_score (27.1% importance)
        # and its imputer has add_indicator=False, so a missing value becomes
        # the training median 25.3 and is indistinguishable from a real 25.3.
        # If coverage moves, the model's behaviour moves with it, silently.
        if "intent_score" in frame.columns and len(frame):
            coverage = float(frame.intent_score.notna().mean())
            baseline = scoring.VENDOR_COVERAGE_TRAINING
            standard_error = math.sqrt(baseline * (1 - baseline) / len(frame))
            sigma = (coverage - baseline) / standard_error if standard_error else 0.0
            detail = (f"coverage {coverage:.2%} vs training baseline {baseline:.2%} "
                      f"({sigma:+.1f} sigma, batch n={len(frame)})")
            if abs(sigma) >= COVERAGE_HALT_SIGMA:
                add("halt", "vendor_coverage_shift", detail,
                    coverage=round(coverage, 4), sigma=round(sigma, 2))
            elif abs(sigma) >= COVERAGE_WARN_SIGMA:
                add("warn", "vendor_coverage_drift", detail,
                    coverage=round(coverage, 4), sigma=round(sigma, 2))
            else:
                add("info", "vendor_coverage_normal", detail,
                    coverage=round(coverage, 4), sigma=round(sigma, 2))

        # --- operational: staleness -----------------------------------------
        if "snapshot_date" in frame.columns and len(frame):
            median_age = float(scoring.snapshot_age_days(frame.snapshot_date).median())
            detail = f"median snapshot age {median_age:.0f} days against 2026-08-01"
            if median_age > STALE_MEDIAN_AGE_DAYS:
                add("warn", "stale_snapshots",
                    detail + " - reps would be calling on year-old information")
            else:
                add("info", "snapshot_freshness_normal", detail)

        halt = any(f["severity"] == "halt" for f in findings)
        gate = {
            "rows_in": len(frame),
            "findings": findings,
            "halt": halt,
            "counts": {
                s: sum(1 for f in findings if f["severity"] == s)
                for s in ("halt", "warn", "info")
            },
        }
        span.outputs = {"halt": halt, "counts": gate["counts"]}

        # Carry the frame forward so the file is read exactly once per run.
        return {"gate": gate, "halted": halt, "accounts": frame}


def route_after_gate(state: AgentState) -> str:
    """The graph's only branch. Deterministic: reads a boolean the gate set.

    Worth being explicit that no LLM participates in this decision. Whether a
    batch is fit to score is a reproducible property of the data, and it is the
    one decision in this pipeline that must never vary between two runs on the
    same input.
    """
    return "halt" if state.get("halted") else "proceed"


# ============================================================================
# 2-4. score -> tier -> flag
# ============================================================================

def score_accounts(state: AgentState) -> dict[str, Any]:
    """Attach P(converts within 90d) to every account.

    Separate from tiering because they fail differently: this node breaks
    loudly if the model or its inputs are wrong, whereas a bad threshold
    produces a perfectly valid list that is quietly pointed at the wrong
    accounts.
    """
    with TRACER.span("score_accounts", run_type="chain") as span:
        model, positive_index = scoring.load_model()
        frame = state["accounts"].copy()
        frame["p"] = scoring.score_frame(model, positive_index, frame)
        span.inputs = {"rows": len(frame), "positive_class_index": positive_index}
        span.outputs = {"p_min": round(float(frame.p.min()), 4),
                        "p_max": round(float(frame.p.max()), 4),
                        "p_mean": round(float(frame.p.mean()), 4)}
        return {"accounts": frame}


def assign_tiers(state: AgentState) -> dict[str, Any]:
    """Convert probabilities into the four buckets a rep actually acts on.

    WHY TIERS AND NOT THE NUMBER: the model ranks well and calibrates badly.
    Its top decile predicts 13.9% and converts 26.7%. Handing a rep '0.1082'
    invites them to read it as a likelihood, which it is not. A/B/C/D says
    only what the model can support - relative ordering.
    """
    with TRACER.span("assign_tiers", run_type="chain") as span:
        frame = state["accounts"].copy()
        frame["tier"] = scoring.assign_tier(frame.p)
        frame = frame.sort_values("p", ascending=False).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        counts = frame.tier.value_counts().to_dict()
        span.inputs = {"thresholds": scoring.TIER_THRESHOLDS}
        span.outputs = {"tier_counts": counts}
        return {"accounts": frame}


def flag_vendor_gap(state: AgentState) -> dict[str, Any]:
    """Mark accounts the model is known to over-score, and say by how much.

    THE RISK: the pipeline imputes a missing intent_score to 25.3 with
    add_indicator=False, so it cannot distinguish 'no vendor data' from 'vendor
    says 25.3'. Measured consequence - accounts with no coverage are scored
    1.61pp above their actual 3.94% conversion rate, against a 6.50% baseline.

    The model cannot be retrained, so the fix is disclosure rather than
    correction: carry the flag all the way to the rep instead of letting a
    clean-looking ranking absorb it.
    """
    with TRACER.span("flag_vendor_gap", run_type="chain") as span:
        frame = state["accounts"].copy()
        frame["has_intent"] = frame.intent_score.notna()
        frame["snapshot_age_days"] = scoring.snapshot_age_days(frame.snapshot_date)
        flagged = int((~frame.has_intent).sum())
        span.outputs = {"accounts_without_vendor_data": flagged,
                        "share": round(flagged / len(frame), 4)}
        return {"accounts": frame}


# ============================================================================
# 5. select_call_list
# ============================================================================

def select_call_list(state: AgentState) -> dict[str, Any]:
    """Cut the ranked list to what one rep can actually work.

    An ordering nobody can finish is still a guess about where to stop, so the
    cut is part of the product rather than a display detail. Capacity is a
    parameter with a stated default because rep headcount and call volume are
    absent from the provided data - it is an assumption, not a finding.

    Accounts are taken in rank order and never above Tier C: tiers C and D
    convert at 4.04% and 2.83%, both below the 6.50% baseline, so padding a
    short list with them would actively waste a rep's day.
    """
    with TRACER.span("select_call_list", run_type="chain") as span:
        frame = state["accounts"]
        capacity = state["capacity"]
        eligible = frame[frame.tier.isin(["A", "B"])]
        selected = eligible.head(capacity).copy()
        span.inputs = {"capacity": capacity, "eligible_AB": len(eligible)}
        span.outputs = {"selected": len(selected),
                        "tier_mix": selected.tier.value_counts().to_dict(),
                        "short_of_capacity": max(0, capacity - len(selected))}
        return {"call_list": selected}


# ============================================================================
# 6-7. generate rationale (LLM leaf) -> validate it (guardrail)
# ============================================================================

def _facts_for(row: pd.Series) -> dict[str, Any]:
    """The only account data the LLM ever sees.

    account_id and the raw probability are deliberately withheld: the id gives
    the model nothing and invites invented specifics, and the probability is
    the one number that must never be echoed back to a rep.
    """
    return {
        "account_type": row.account_type,
        "industry": row.industry,
        "employee_count": int(row.employee_count),
        "has_intent": bool(row.has_intent),
        "intent_score": None if pd.isna(row.intent_score) else float(row.intent_score),
        "mql_count_90d": int(row.mql_count_90d),
        "web_touchpoints_90d": int(row.web_touchpoints_90d),
        "sales_contacts_90d": int(row.sales_contacts_90d),
        "trial_started": int(row.trial_started),
        "trial_active_users": int(row.trial_active_users),
        "snapshot_age_days": int(row.snapshot_age_days),
    }


def generate_rationales(state: AgentState) -> dict[str, Any]:
    """Ask the (mocked) LLM for a readable justification per selected account.

    This node makes no decision. Ranking, tiering, flagging and the capacity
    cut all happened upstream in deterministic code; the LLM writes prose about
    a list that is already final. That containment is the design, not an
    accident of scope - see agent/mocks.py for the full reasoning.
    """
    provider = state["provider"]
    with TRACER.span("generate_rationales", run_type="chain",
                     provider=provider.describe()) as parent:
        briefs: list[dict[str, Any]] = []
        for _, row in state["call_list"].iterrows():
            facts = _facts_for(row)
            with TRACER.span("llm.rationale", run_type="llm",
                             account_id=row.account_id,
                             provider=provider.name, model=provider.model) as span:
                span.inputs = facts
                try:
                    payload = provider.generate(facts)
                    source = "llm"
                except LLMCallFailed as exc:
                    # One account failing does not fail the run. The provider
                    # itself was already validated at startup, so this is a
                    # per-call problem (rate limit, malformed reply, timeout).
                    payload = render_fallback_rationale(facts)
                    source = "fallback_llm_error"
                    span.outputs = {"error": str(exc)}
                else:
                    span.outputs = payload
            briefs.append({"account_id": row.account_id, "facts": facts,
                           "payload": payload, "source": source})
        parent.outputs = {"generated": len(briefs)}
        return {"briefs": briefs}


def validate_rationales(state: AgentState) -> dict[str, Any]:
    """Check every rationale before a rep can see it; substitute the template if not.

    WHY THIS IS A NODE AND NOT A HELPER: rejecting LLM output is a step in the
    pipeline with its own success rate, and that rate is a monitored signal. A
    quiet climb in guardrail rejections is an early warning that prompt, model
    or data have moved - visible here, invisible if this were an inline if.
    """
    with TRACER.span("validate_rationales", run_type="chain") as parent:
        checked: list[dict[str, Any]] = []
        rejected = 0
        for brief in state["briefs"]:
            with TRACER.span("guardrail.validate_rationale", run_type="evaluator",
                             account_id=brief["account_id"]) as span:
                result = validate_rationale(brief["payload"], brief["facts"])
                span.inputs = brief["payload"]
                span.outputs = {"ok": result.ok, "failures": result.codes()}
                if not result.ok:
                    rejected += 1
                    brief = {**brief,
                             "payload": render_fallback_rationale(brief["facts"]),
                             "source": "fallback_guardrail_rejected",
                             "guardrail_failures": result.failures}
            checked.append(brief)
        parent.outputs = {"checked": len(checked), "rejected": rejected}
        return {"briefs": checked}


# ============================================================================
# 8. render_outputs  /  halt_run
# ============================================================================

def _run_report(state: AgentState, briefs: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Assemble the observability artifact the monitoring work consumes."""
    frame = state.get("accounts")
    report: dict[str, Any] = {
        "trace": TRACER.summary(),
        "input": {"path": state["input_csv"], "rows": state["gate"]["rows_in"]},
        "gate": state["gate"],
        "halted": bool(state.get("halted")),
    }
    if frame is not None and "tier" in frame.columns:
        report["scoring"] = {
            "tier_counts": frame.tier.value_counts().sort_index().to_dict(),
            "tier_thresholds": scoring.TIER_THRESHOLDS,
            "score_distribution": {
                k: round(float(v), 4) for k, v in
                frame.p.describe(percentiles=[.5, .9]).to_dict().items()
            },
            "accounts_without_vendor_data": int((~frame.has_intent).sum()),
        }
    if briefs is not None:
        sources = pd.Series([b["source"] for b in briefs]).value_counts().to_dict()
        failures: dict[str, int] = {}
        for brief in briefs:
            for failure in brief.get("guardrail_failures", []):
                failures[failure["code"]] = failures.get(failure["code"], 0) + 1
        provider = state.get("provider")
        report["llm"] = {
            # Recorded per run so an artifact can never misrepresent which
            # backend produced it. A mock run says so; a live run names the model.
            "provider": provider.name if provider else "unknown",
            "model": provider.model if provider else "unknown",
            "mocked": (not provider.is_live) if provider else True,
            "briefs": len(briefs),
            "by_source": sources,
            "guardrail_rejections": sum(
                1 for b in briefs if b["source"] == "fallback_guardrail_rejected"),
            "guardrail_failure_codes": failures,
        }
    return report


def _render_call_sheet(state: AgentState, briefs: list[dict[str, Any]]) -> str:
    """The artifact a rep actually opens. Markdown so it pastes anywhere."""
    frame = state["call_list"].set_index("account_id")
    gate_warnings = [f for f in state["gate"]["findings"] if f["severity"] == "warn"]

    lines = [
        "# Call list",
        "",
        f"{len(briefs)} accounts, highest priority first. "
        f"Ordering comes from a conversion model; treat it as an ordering, not a forecast.",
        "",
    ]
    if gate_warnings:
        lines += ["> **Data warnings on this batch**"] + \
                 [f"> - {w['detail']}" for w in gate_warnings] + [""]

    for position, brief in enumerate(briefs, start=1):
        row = frame.loc[brief["account_id"]]
        payload = brief["payload"]
        lines += [
            f"## {position}. {brief['account_id']} — Tier {row.tier}",
            "",
            f"*{row.account_type} · {row.industry} · ~{int(row.employee_count)} staff · "
            f"data {int(row.snapshot_age_days)} days old*",
            "",
            payload["rationale"],
            "",
            f"**Try opening with:** {payload['opening_line']}",
        ]
        if payload.get("data_caveat"):
            lines += ["", f"> ⚠ {payload['data_caveat']}"]
        if brief["source"] != "llm":
            lines += ["", f"> *(summary written by fallback template: {brief['source']})*"]
        lines.append("")
    return "\n".join(lines)


def _crm_tasks(state: AgentState, briefs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Salesforce task payloads - the action attached to the list.

    MOCKED AT THIS BOUNDARY: no Salesforce credentials exist for this exercise,
    so the agent writes the payloads it would POST rather than posting them.
    A real implementation swaps this for a bulk create against the Tasks
    endpoint, keyed on account_id, idempotent on (account_id, run date) so a
    re-run does not duplicate a rep's queue.
    """
    frame = state["call_list"].set_index("account_id")
    tasks = []
    for brief in briefs:
        row = frame.loc[brief["account_id"]]
        payload = brief["payload"]
        description = payload["rationale"]
        if payload.get("data_caveat"):
            description += f" NOTE: {payload['data_caveat']}"
        tasks.append({
            "sobject": "Task",
            "WhatId": brief["account_id"],
            "Subject": f"[Tier {row.tier}] Call {brief['account_id']}",
            "Description": description,
            "Suggested_Opening__c": payload["opening_line"],
            "Priority": {"A": "High", "B": "Normal"}.get(row.tier, "Low"),
            "Status": "Not Started",
            "Vendor_Intent_Data_Held__c": bool(row.has_intent),
            "Rationale_Source__c": brief["source"],
        })
    return tasks


def render_outputs(state: AgentState) -> dict[str, Any]:
    """Write the three artifacts and record where they went."""
    with TRACER.span("render_outputs", run_type="chain") as span:
        out = Path(state["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        briefs = state["briefs"]

        paths = {
            "call_sheet": out / "call_sheet.md",
            "crm_tasks": out / "crm_tasks.json",
            "run_report": out / "run_report.json",
            "scored_accounts": out / "agent_scored_accounts.csv",
        }
        paths["call_sheet"].write_text(_render_call_sheet(state, briefs), encoding="utf-8")
        paths["crm_tasks"].write_text(
            json.dumps(_crm_tasks(state, briefs), indent=2), encoding="utf-8")

        columns = ["rank", "account_id", "tier", "p", "has_intent", "account_type",
                   "industry", "employee_count", "intent_score", "snapshot_age_days"]
        state["accounts"][columns].to_csv(paths["scored_accounts"], index=False)

        report = _run_report(state, briefs)
        report["artifacts"] = {k: str(v) for k, v in paths.items()}
        paths["run_report"].write_text(json.dumps(report, indent=2), encoding="utf-8")

        span.outputs = {k: str(v) for k, v in paths.items()}
        return {"artifacts": {k: str(v) for k, v in paths.items()}, "report": report}


def halt_run(state: AgentState) -> dict[str, Any]:
    """Write the diagnosis and stop. Deliberately produces no call list.

    Emitting a degraded list here would be the worse failure. A rep who is
    handed nothing asks why; a rep handed a subtly wrong list works it, and
    nobody finds out for a quarter.
    """
    with TRACER.span("halt_run", run_type="chain") as span:
        out = Path(state["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        report = _run_report(state, briefs=None)
        report["outcome"] = "HALTED - input failed validation, no call list produced"
        path = out / "run_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        span.outputs = {"run_report": str(path)}
        return {"artifacts": {"run_report": str(path)}, "report": report}
