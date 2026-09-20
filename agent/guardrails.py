"""
Content validation for LLM-written rationales, plus the self-test that proves
the checks actually fire.

WHY THIS EXISTS: the structured-output schema guarantees the SHAPE of the
response and says nothing about whether the content is true or safe to show a
rep. The three failures that matter here are all shape-valid:

  1. An invented fact. The model adds a number, headcount or event that was
     never in the prompt. A rep repeats it on a call and is wrong in front of
     a customer. This is the expensive one.
  2. Likelihood language. Section 5-7 of this project established the model
     ranks well and calibrates badly - the top decile predicts 13.9% and
     actually converts 26.7%. Any sentence implying an account "will convert"
     turns a ranking into a promise the numbers do not support.
  3. A dropped caveat. 40% of accounts have no third-party intent data and the
     model over-scores them by 1.61pp. If that disclosure silently disappears
     from the rendered output, the blind spot is hidden exactly where it
     matters.

Every check below maps to one of those. Run the harness directly:

    python -m agent.guardrails
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Phrases that turn a ranking into a prediction. Matched case-insensitively as
# substrings, so "will convert" also catches "they will convert soon".
LIKELIHOOD_PHRASES = [
    "will convert", "will close", "will buy", "will sign",
    "likely to convert", "likely to buy", "likely to close",
    "guaranteed", "sure thing", "strong bet", "safe bet",
    "high probability", "high chance", "certain to", "expect them to buy",
]

# Internal vocabulary that should never reach a rep. Word-boundary matched so
# "scored" trips but "underscore" does not.
ML_VOCABULARY = [
    "model", "score", "scored", "scoring", "probability", "percentile",
    "tier", "algorithm", "prediction", "predicted", "ranking model",
    "machine learning", "confidence interval",
]

# Numbers that may appear without being in the account facts. 90 is the fixed
# reporting window every activity field uses ("last 90 days").
ALWAYS_ALLOWED_NUMBERS = {90.0}

MAX_RATIONALE_SENTENCES = 2


@dataclass
class ValidationResult:
    """Outcome for one rationale. `failures` is empty iff the output is usable."""
    ok: bool
    failures: list[dict[str, str]] = field(default_factory=list)

    def codes(self) -> list[str]:
        return [f["code"] for f in self.failures]


def _fail(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _numbers_in(text: str) -> list[float]:
    """Every numeric literal in the text, as floats.

    Strips thousands separators first so '1,543' reads as one number rather
    than two, which would otherwise produce a false invented-number failure.
    """
    return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]


def _allowed_numbers(facts: dict[str, Any]) -> set[float]:
    """Numeric values the model was actually given, so anything else is invented."""
    allowed = set(ALWAYS_ALLOWED_NUMBERS)
    for key in ("employee_count", "mql_count_90d", "web_touchpoints_90d",
                "sales_contacts_90d", "trial_active_users", "snapshot_age_days"):
        value = facts.get(key)
        if value is not None:
            allowed.add(float(value))
    if facts.get("has_intent") and facts.get("intent_score") is not None:
        allowed.add(float(facts["intent_score"]))
    return allowed


def validate_rationale(payload: Any, facts: dict[str, Any]) -> ValidationResult:
    """Check one LLM rationale against the account facts it was generated from.

    Returns every failure rather than stopping at the first, so the run report
    shows the full picture of how a bad response was bad.
    """
    failures: list[dict[str, str]] = []

    # --- shape -------------------------------------------------------------
    # Cheap, but not redundant with the schema: the fallback path and any
    # future non-schema-enforced provider both come through here too.
    if not isinstance(payload, dict):
        return ValidationResult(False, [_fail("not_an_object", f"got {type(payload).__name__}")])
    for key in ("rationale", "opening_line", "data_caveat"):
        if key not in payload:
            failures.append(_fail("missing_field", key))
    for key in ("rationale", "opening_line"):
        value = payload.get(key)
        if key in payload and (not isinstance(value, str) or not value.strip()):
            failures.append(_fail("empty_field", key))
    if failures:
        return ValidationResult(False, failures)

    rationale = payload["rationale"]
    opening = payload["opening_line"]
    caveat = payload["data_caveat"]
    prose = " ".join(filter(None, [rationale, opening, caveat]))
    lowered = prose.lower()

    # --- length ------------------------------------------------------------
    # A rep skims this. Three sentences means it will not get read.
    sentences = [s for s in re.split(r"[.!?]+", rationale) if s.strip()]
    if len(sentences) > MAX_RATIONALE_SENTENCES:
        failures.append(_fail(
            "too_long", f"{len(sentences)} sentences, max {MAX_RATIONALE_SENTENCES}"))

    # --- failure mode 2: likelihood language -------------------------------
    for phrase in LIKELIHOOD_PHRASES:
        if phrase in lowered:
            failures.append(_fail("likelihood_language", f"contains {phrase!r}"))

    # --- internal vocabulary -----------------------------------------------
    for term in ML_VOCABULARY:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            failures.append(_fail("ml_vocabulary", f"contains {term!r}"))

    # --- failure mode 1: invented facts ------------------------------------
    allowed = _allowed_numbers(facts)
    for number in _numbers_in(prose):
        if not any(abs(number - a) < 1e-6 for a in allowed):
            failures.append(_fail(
                "invented_number",
                f"{number:g} is not among the facts supplied ({sorted(allowed)})"))

    # --- failure mode 3: the vendor-data caveat ----------------------------
    if not facts.get("has_intent"):
        if not (isinstance(caveat, str) and caveat.strip()):
            failures.append(_fail(
                "missing_required_caveat",
                "no third-party intent data for this account, so the caveat is mandatory"))
        elif "intent" not in caveat.lower():
            failures.append(_fail(
                "caveat_does_not_mention_intent_data", caveat[:80]))
    elif isinstance(caveat, str) and caveat.strip():
        # A caveat on a covered account means the model misread its input.
        failures.append(_fail("unnecessary_caveat", caveat[:80]))

    return ValidationResult(not failures, failures)


# ============================================================================
# SELF-TEST HARNESS
# ============================================================================
# A guardrail nobody has watched fail is indistinguishable from one that always
# returns True. These cases exercise each check in both directions.

_COVERED = {
    "account_type": "Prospect", "industry": "Retail", "employee_count": 188,
    "has_intent": True, "intent_score": 33.2, "mql_count_90d": 1,
    "web_touchpoints_90d": 5, "sales_contacts_90d": 2,
    "trial_started": 1, "trial_active_users": 3, "snapshot_age_days": 290,
}
_UNCOVERED = {**_COVERED, "has_intent": False, "intent_score": None,
              "employee_count": 637, "trial_started": 0, "trial_active_users": 0}

_CASES: list[tuple[str, dict, dict, list[str]]] = [
    (
        "clean output on a covered account",
        _COVERED,
        {"rationale": "This prospect is worth a look because 3 people are already active in a trial.",
         "opening_line": "Ask how the trial has been going.",
         "data_caveat": None},
        [],
    ),
    (
        "clean output on an uncovered account, caveat present",
        _UNCOVERED,
        {"rationale": "This prospect has had 5 website visits in the last 90 days.",
         "opening_line": "Introduce yourself and ask about their workflow.",
         "data_caveat": "We hold no third-party intent data on this account, so its "
                        "position on the list is less reliable."},
        [],
    ),
    (
        "caveat dropped on an uncovered account",
        _UNCOVERED,
        {"rationale": "This prospect has had 5 website visits in the last 90 days.",
         "opening_line": "Introduce yourself.",
         "data_caveat": None},
        ["missing_required_caveat"],
    ),
    (
        "invented headcount",
        _COVERED,
        {"rationale": "This 450-person retailer has 5 website visits in the last 90 days.",
         "opening_line": "Ask what prompted the interest.",
         "data_caveat": None},
        ["invented_number"],
    ),
    (
        "promises an outcome",
        _COVERED,
        {"rationale": "With 3 trial users this account will convert shortly.",
         "opening_line": "Ask how the trial is going.",
         "data_caveat": None},
        ["likelihood_language"],
    ),
    (
        "leaks internal vocabulary",
        _COVERED,
        {"rationale": "The model scored this account in the top tier.",
         "opening_line": "Ask how the trial is going.",
         "data_caveat": None},
        ["ml_vocabulary"],
    ),
    (
        "rationale runs long",
        _COVERED,
        {"rationale": "They started a trial. Three people used it. They also visited the site. "
                      "Worth a call.",
         "opening_line": "Ask how the trial is going.",
         "data_caveat": None},
        ["too_long"],
    ),
    (
        "caveat attached to a covered account",
        _COVERED,
        {"rationale": "This prospect has 3 active trial users.",
         "opening_line": "Ask how the trial is going.",
         "data_caveat": "No third-party intent data is held for this account."},
        ["unnecessary_caveat"],
    ),
    (
        "malformed payload",
        _COVERED,
        {"rationale": "", "opening_line": "Call them.", "data_caveat": None},
        ["empty_field"],
    ),
]


def run_self_test() -> tuple[int, int]:
    """Run every case. Returns (passed, total) and prints a per-case verdict."""
    passed = 0
    print(f"{'case':<48}{'expected':<34}{'verdict'}")
    print("-" * 100)
    for name, facts, payload, expected_codes in _CASES:
        result = validate_rationale(payload, facts)
        actual = result.codes()
        if expected_codes:
            success = all(code in actual for code in expected_codes)
            expectation = "rejects: " + ",".join(expected_codes)
        else:
            success = result.ok
            expectation = "accepts"
        passed += success
        verdict = "PASS" if success else f"FAIL (got {actual or 'accepted'})"
        print(f"{name:<48}{expectation:<34}{verdict}")
    print("-" * 100)
    print(f"{passed}/{len(_CASES)} guardrail checks behaved as specified")
    return passed, len(_CASES)


if __name__ == "__main__":
    import sys
    ok, total = run_self_test()
    sys.exit(0 if ok == total else 1)
