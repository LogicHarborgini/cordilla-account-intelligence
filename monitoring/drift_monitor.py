"""
Leading-indicator drift monitor. Reads a history of agent runs and answers a
question no single run can: is this system sliding?

    python -m monitoring.drift_monitor --simulate   # demo on a generated history
    python -m monitoring.drift_monitor --history output/run_history/

WHY THIS EXISTS SEPARATELY FROM THE AGENT'S INPUT GATE
------------------------------------------------------
The gate in agent/nodes.py inspects one batch and decides whether to score it.
It is a threshold on a single observation, and it works: a coverage collapse to
18% trips it at -14.8 sigma and the run halts.

It is also structurally blind to the failure this project is actually about.
The scenario describes a scoring system that looked right at launch and lost
credibility a couple of quarters later, with nobody able to point at the moment
it broke. That is not a cliff. That is a slow slide where every individual run
sits comfortably inside tolerance.

Concretely: vendor coverage on a 300-account batch has a sampling standard
error of about 2.8pp. A 3-sigma gate therefore fires at roughly 8.4pp of
movement. A vendor quietly shedding 0.8pp of coverage per weekly run takes
about ten weeks to reach that - and the gate is silent for every single one of
those runs while the model's most important feature degrades underneath it.

So this module watches the same signals ACROSS runs, using a method built for
exactly that: CUSUM.

WHY CUSUM RATHER THAN A TREND LINE OR A WIDER THRESHOLD
-------------------------------------------------------
CUSUM accumulates small deviations from a known target instead of testing each
one in isolation, which makes it sensitive to a persistent small shift while
staying insensitive to one-off noise - the precise tradeoff needed here. A
regression slope would also detect the trend, but it needs a window length
chosen in advance and it re-answers the question from scratch each time; CUSUM
carries state, so a drift that starts today is detected on its own schedule
rather than whenever the window happens to align.

Parameters are the textbook ones: k = 0.5 (slack, in sigma - ignore deviations
smaller than half a sigma) and h = 5.0 (decision interval). That pairing
detects a sustained 1-sigma shift within a handful of observations while
producing a false alarm roughly once in 465 in-control runs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --- CUSUM tuning -----------------------------------------------------------
CUSUM_SLACK_K = 0.5
CUSUM_THRESHOLD_H = 5.0

# --- Point-check tuning, mirroring the agent's input gate -------------------
POINT_WARN_SIGMA = 3.0
POINT_HALT_SIGMA = 6.0

# --- Noise estimation -------------------------------------------------------
# How many leading runs are treated as a known-good period for estimating how
# much each signal moves when nothing is wrong.
#
# WHY THIS IS NOT OPTIONAL. The theoretical sampling error assumes every batch
# is a random draw from one fixed population. Real batches are not: territories
# rotate, campaigns land, exports change shape. So sampling error is a FLOOR on
# the noise, not an estimate of it, and using it alone makes a monitor
# hypersensitive - the mean predicted probability on n=300 has a standard error
# of 0.0018, so an entirely unremarkable 0.7% population shift reads as 4 sigma
# and pages somebody every week. An alarm that cries wolf gets muted, and a
# muted alarm is how the original Cordilla effort went unwatched.
#
# So: take the larger of the theoretical floor and what the signal actually did
# during the burn-in. The burn-in must be a period believed to be healthy; if
# drift is already underway during it, the estimated noise is inflated and the
# monitor goes quiet. That is a real limitation and the reason a re-baseline
# should always be a deliberate, recorded act.
BURN_IN_RUNS = 6


@dataclass
class Signal:
    """One monitored quantity: where it comes from, and what normal means.

    `target` and `sigma_of` are what make this a monitor rather than a log.
    Without a stated expectation and a stated noise model there is no way to
    separate "moved" from "moved more than it should have".
    """
    key: str
    label: str
    target: float
    direction: str                 # "down" | "up" | "both" - which way is bad
    sigma_of: Any                  # callable(run) -> float, the noise scale
    provenance: str                # where the target came from
    consequence: str               # what it means for the product if this moves


def _binomial_se(p: float, n: int) -> float:
    """Sampling noise for a proportion. Scales with batch size, as it should."""
    return math.sqrt(p * (1 - p) / n) if n else float("inf")


# The signals, in the order a responder should read them. Every target traces
# to a measured figure from the analysis scripts, not to a guess.
SIGNALS: list[Signal] = [
    Signal(
        key="vendor_coverage",
        label="Third-party intent data coverage",
        target=0.5983,
        direction="both",
        sigma_of=lambda run: _binomial_se(0.5983, run["rows"]),
        provenance="718/1200 of training rows carry an intent_score "
                   "(analysis/explore_data.py)",
        consequence="intent_score is the model's top feature at 27.1% importance and "
                    "its imputer has add_indicator=False, so a missing value silently "
                    "becomes the training median 25.3. If coverage moves, what the "
                    "model is actually reading moves with it, with no error raised.",
    ),
    Signal(
        key="tier_a_share",
        label="Share of batch reaching Tier A",
        target=0.10,
        direction="down",
        sigma_of=lambda run: _binomial_se(0.10, run["rows"]),
        provenance="Tier A cutoff is the training p90 by construction, so a batch "
                   "resembling the training population yields ~10%",
        consequence="This is free drift detection bought by using frozen cutoffs "
                    "instead of per-batch percentiles. A shrinking Tier A means the "
                    "population genuinely got weaker; percentile tiering would have "
                    "hidden it by always labelling a top 10%.",
    ),
    Signal(
        key="score_mean",
        label="Mean predicted probability",
        target=0.0661,
        direction="both",
        # Dispersion of the score distribution itself, not sampling noise on a
        # proportion, so the observed std is the right scale here.
        sigma_of=lambda run: run.get("score_std", 0.0316) / math.sqrt(max(run["rows"], 1)),
        provenance="mean of predict_proba over the 1,200 training rows "
                   "(analysis/score_and_evaluate.py)",
        consequence="Moves if the input population shifts or the model is swapped. "
                    "Ambiguous on its own - read alongside coverage to tell a data "
                    "change from a model change.",
    ),
    Signal(
        key="guardrail_rejection_rate",
        label="LLM rationales rejected by the guardrail",
        target=0.0,
        direction="up",
        sigma_of=lambda run: max(_binomial_se(0.02, max(run.get("briefs", 1), 1)), 0.01),
        provenance="0 rejections across 25 briefs on the current mocked implementation",
        consequence="A climbing rejection rate means the model, the prompt or the "
                    "input distribution moved. It is the earliest available warning "
                    "on the LLM half, and it costs nothing - the fallback template "
                    "already keeps reps supplied while it is investigated.",
    ),
]


@dataclass
class SignalState:
    """Running CUSUM state for one signal across the whole history."""
    signal: Signal
    cusum_high: float = 0.0
    cusum_low: float = 0.0
    fired_at: int | None = None
    point_alerts: list[int] = field(default_factory=list)
    series: list[tuple[int, float, float]] = field(default_factory=list)  # (idx, value, z)
    sigma_used: float = 0.0
    sigma_source: str = "theoretical"


def _empirical_sigma(runs: list[dict[str, Any]], key: str) -> float | None:
    """Observed spread of a signal during the burn-in, or None if too short."""
    values = [r[key] for r in runs[:BURN_IN_RUNS] if key in r]
    if len(values) < 3:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def evaluate(runs: list[dict[str, Any]]) -> dict[str, SignalState]:
    """Walk the history in order, updating point checks and CUSUM per signal."""
    states = {s.key: SignalState(signal=s) for s in SIGNALS}

    # Fix the noise scale once, from the burn-in, so it cannot creep upward as
    # drift arrives and quietly desensitise the monitor against itself.
    for signal in SIGNALS:
        observed = _empirical_sigma(runs, signal.key)
        theoretical = signal.sigma_of(runs[0]) if runs else 0.0
        state = states[signal.key]
        if observed is not None and observed > theoretical:
            state.sigma_used, state.sigma_source = observed, f"burn-in ({BURN_IN_RUNS} runs)"
        else:
            state.sigma_used, state.sigma_source = theoretical, "theoretical floor"

    for index, run in enumerate(runs, start=1):
        for signal in SIGNALS:
            if signal.key not in run:
                continue
            state = states[signal.key]
            value = run[signal.key]
            sigma = state.sigma_used or signal.sigma_of(run)
            z = (value - signal.target) / sigma if sigma else 0.0
            state.series.append((index, value, z))

            # --- point check: would the per-run gate have said anything? ----
            if abs(z) >= POINT_WARN_SIGMA:
                state.point_alerts.append(index)

            # --- CUSUM: has a small deviation persisted? --------------------
            state.cusum_high = max(0.0, state.cusum_high + z - CUSUM_SLACK_K)
            state.cusum_low = max(0.0, state.cusum_low - z - CUSUM_SLACK_K)

            breached = (
                (signal.direction in ("up", "both") and state.cusum_high >= CUSUM_THRESHOLD_H)
                or (signal.direction in ("down", "both") and state.cusum_low >= CUSUM_THRESHOLD_H)
            )
            if breached and state.fired_at is None:
                state.fired_at = index
    return states


# ============================================================================
# WHAT HAPPENS WHEN IT TRIPS
# ============================================================================
# A check with no defined response is a log line. Each severity below names an
# owner, an action, and - critically - whether reps keep getting a list.

RESPONSE_PLAYBOOK = {
    "point_halt": {
        "meaning": "A single batch is structurally wrong (>=6 sigma).",
        "automatic": "The agent's input gate already halted the run. No call list "
                     "was produced and no CRM tasks were created.",
        "human": "Page the owning engineer. Reps fall back to their existing "
                 "prioritisation for the day - this is one day, not a quarter.",
        "resolve": "Do not re-run until the input is explained. A halt that is "
                   "cleared by re-running is the failure mode returning.",
    },
    "cusum_fired": {
        "meaning": "A small deviation has persisted long enough that it is not "
                   "noise. THIS IS THE ONE THE BACKSTORY IS ABOUT - no single run "
                   "ever looked wrong.",
        "automatic": "Runs continue. The call sheet carries a data-quality banner "
                     "so reps see the caveat at the point of use, not in a dashboard "
                     "nobody opens.",
        "human": "Open a ticket against the owning engineer within one working day. "
                 "Establish whether the vendor changed coverage, the upstream export "
                 "changed, or the population genuinely moved.",
        "resolve": "Either fix the input, or re-baseline the targets deliberately "
                   "and record why. Re-baselining silently is how the original "
                   "Cordilla effort lost its credibility.",
    },
    "point_warn": {
        "meaning": "One batch is unusual (>=3 sigma) but not structurally broken.",
        "automatic": "Runs continue, warning recorded in run_report.json and shown "
                     "on the call sheet.",
        "human": "No page. Review at the next weekly check unless CUSUM also fires.",
        "resolve": "Usually self-resolving. Two in a row is a CUSUM precursor.",
    },
}


def render(states: dict[str, SignalState], runs: list[dict[str, Any]]) -> int:
    """Print the report. Returns a shell exit code: 0 clean, 1 something fired."""
    print("=" * 78)
    print(f"DRIFT MONITOR — {len(runs)} runs")
    print("=" * 78)

    any_fired = False
    for signal in SIGNALS:
        state = states[signal.key]
        if not state.series:
            continue
        first, last = state.series[0], state.series[-1]
        print(f"\n{signal.label}")
        print(f"  target {signal.target:.4f}   ({signal.provenance})")
        print(f"  noise  sigma {state.sigma_used:.4f} from {state.sigma_source}")
        print(f"  first run {first[1]:.4f} ({first[2]:+.1f} sigma)"
              f"  ->  latest {last[1]:.4f} ({last[2]:+.1f} sigma)")
        print(f"  CUSUM  high {state.cusum_high:.2f}  low {state.cusum_low:.2f}"
              f"   (fires at {CUSUM_THRESHOLD_H})")

        if state.point_alerts:
            print(f"  POINT CHECK fired on run(s) {state.point_alerts}")
        else:
            print(f"  point check: never fired "
                  f"(no single run reached {POINT_WARN_SIGMA:.0f} sigma)")

        if state.fired_at is not None:
            any_fired = True
            print(f"  >> CUSUM ALERT at run {state.fired_at}: sustained drift")
            if not state.point_alerts:
                print("     Caught only because runs are compared to each other.")
                print("     Every individual run passed its own threshold.")
            print(f"     WHY IT MATTERS: {signal.consequence}")
        else:
            print("  >> in control")

    print("\n" + "=" * 78)
    if any_fired:
        play = RESPONSE_PLAYBOOK["cusum_fired"]
        print("RESPONSE — sustained drift detected")
        print("=" * 78)
        for field_name in ("meaning", "automatic", "human", "resolve"):
            print(f"  {field_name:<10} {play[field_name]}")
    else:
        print("All monitored signals in control.")
    return 1 if any_fired else 0


# ============================================================================
# DEMONSTRATION HISTORY
# ============================================================================

def simulate_history(n_stable: int = 6, n_drifting: int = 12,
                     drift_per_run: float = 0.008, seed: int = 7) -> list[dict[str, Any]]:
    """Generate a run history that drifts slowly enough to evade the point check.

    THIS IS SYNTHETIC AND LABELLED AS SUCH. It exists because a detector nobody
    has watched fire is indistinguishable from one that never fires, and a
    single real run cannot exercise a cross-run check.

    Two choices here are deliberate rather than convenient:

    1. Drift of 0.8pp of coverage per run is ~0.28 sigma per step against the
       2.8pp sampling error on n=300, so no individual run comes close to the
       3-sigma gate. The demo asks whether accumulating those steps finds what
       thresholding them one at a time cannot.

    2. The injected run-to-run noise is LARGER than sampling error - 3.5pp on
       coverage against a 2.8pp theoretical floor. That is the realistic case:
       territories rotate and campaigns land, so batches differ by more than
       random sampling would predict. Simulating noise smaller than sampling
       error would quietly make the detector look better than it is, and would
       also stop the burn-in estimator from ever engaging.
    """
    import random
    rng = random.Random(seed)
    runs = []
    for i in range(n_stable + n_drifting):
        drift = max(0, i - n_stable + 1) * drift_per_run
        coverage = 0.5983 - drift + rng.gauss(0, 0.035)
        # Tier A share tracks coverage: losing vendor data pushes accounts onto
        # the imputed median, which pulls scores toward the middle of the range.
        tier_a = max(0.0, 0.10 - drift * 0.55 + rng.gauss(0, 0.020))
        runs.append({
            "run": f"sim-{i + 1:02d}",
            "rows": 300,
            "vendor_coverage": round(coverage, 4),
            "tier_a_share": round(tier_a, 4),
            "score_mean": round(0.0661 - drift * 0.12 + rng.gauss(0, 0.0030), 4),
            "score_std": 0.0316,
            "briefs": 25,
            "guardrail_rejection_rate": 0.0,
        })
    return runs


def load_history(directory: Path) -> list[dict[str, Any]]:
    """Read real run_report.json files into the flat shape the monitor expects.

    In deployment the agent would append one row per run to a store; reading a
    directory of reports keeps the prototype dependency-free while using the
    exact fields the agent already emits.
    """
    runs = []
    for path in sorted(directory.glob("**/run_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        coverage = next(
            (f.get("coverage") for f in report["gate"]["findings"]
             if f["code"].startswith("vendor_coverage")), None)
        row: dict[str, Any] = {"run": path.parent.name,
                               "rows": report["gate"]["rows_in"]}
        if coverage is not None:
            row["vendor_coverage"] = coverage
        scoring_section = report.get("scoring")
        if scoring_section:
            counts = scoring_section["tier_counts"]
            total = sum(counts.values())
            row["tier_a_share"] = counts.get("A", 0) / total if total else 0.0
            row["score_mean"] = scoring_section["score_distribution"]["mean"]
            row["score_std"] = scoring_section["score_distribution"].get("std", 0.0316)
        llm_section = report.get("llm")
        if llm_section and llm_section.get("briefs"):
            row["briefs"] = llm_section["briefs"]
            row["guardrail_rejection_rate"] = (
                llm_section["guardrail_rejections"] / llm_section["briefs"])
        runs.append(row)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--history", type=Path,
                        help="directory containing run_report.json files")
    parser.add_argument("--simulate", action="store_true",
                        help="run against a generated drifting history (synthetic)")
    args = parser.parse_args()

    if args.simulate:
        runs = simulate_history()
        print("NOTE: synthetic history — coverage declining 0.8pp per run, which is")
        print("      ~0.29 sigma per step against a 2.8pp standard error on n=300.")
        print("      Generated to exercise the detector, not presented as data.\n")
    elif args.history:
        runs = load_history(args.history)
        if not runs:
            print(f"no run_report.json found under {args.history}")
            return 2
    else:
        parser.error("pass --history DIR or --simulate")

    return render(evaluate(runs), runs)


if __name__ == "__main__":
    raise SystemExit(main())
