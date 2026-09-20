"""
The lagging check: did the accounts we told reps to call actually convert?

    python -m monitoring.outcome_monitor

This is the only check that tests the claim the product is actually making.
Everything in drift_monitor.py watches inputs and outputs of the scoring step;
none of it can tell you the model became wrong, only that something upstream
moved. This one closes the loop against ground truth.

It also has a conclusion built into it that changes how the rest of the
monitoring should be read, so it is worth stating up front:

    THIS CHECK IS STRUCTURALLY TOO SLOW TO BE THE PRIMARY ALARM.

Not because it is badly designed - because of arithmetic. Tier A is ~28
accounts per batch, conversion is defined over 90 days, and detecting a
moderate drop in a conversion rate needs a few hundred observations. Run the
numbers (this module does) and the time-to-detection for a drop from 26.7% to
20% lands around five months.

Five months is a couple of quarters. That is precisely the interval in the
Cordilla story where scores "stopped matching what reps saw in the field" and
nobody could say when it started. An organisation watching only conversion
outcomes would reproduce that failure exactly, and would be doing nothing
wrong - the instrument is simply too slow.

That is the argument for the leading indicators in drift_monitor.py. They
cannot prove the model is wrong, but they can flag the input change that
causes it, weeks to months before the outcome data could. This check is the
confirmation, not the alarm.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from scipy import stats

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import scoring  # noqa: E402

# The claim under test, measured on the 1,200 labelled training rows by
# analysis/score_and_evaluate.py. IN-SAMPLE, so it is an upper bound: the model
# was fit on these rows and no holdout exists. A live cohort matching it would
# be a pleasant surprise; the check is calibrated to catch a REAL fall, not to
# confirm the number.
TIER_A_EXPECTED_RATE = scoring.TIER_OBSERVED_RATE["A"]   # 0.2667
BASELINE_RATE = scoring.BASELINE_CONVERSION_RATE         # 0.0650

# A drop to here means the top tier no longer earns the rep's time relative to
# the free two-fact heuristic established during the impact work, which reaches
# an in-sample AUC of 0.612 at zero cost. This is the decision boundary that
# matters, not statistical significance for its own sake.
TIER_A_ACTIONABLE_FLOOR = 0.15

ALPHA = 0.05      # one-sided: only a FALL is bad news
POWER = 0.80

TIER_A_PER_BATCH = 28          # observed on data/accounts_to_score.csv
CONVERSION_WINDOW_DAYS = 90    # converted_within_90d, by definition


def required_sample_size(p0: float, p1: float,
                         alpha: float = ALPHA, power: float = POWER) -> int:
    """Tier A accounts needed to detect a fall from p0 to p1.

    One-sided one-sample proportion test. One-sided because a top tier
    performing BETTER than expected is not an incident.
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    numerator = (z_alpha * math.sqrt(p0 * (1 - p0))
                 + z_beta * math.sqrt(p1 * (1 - p1))) ** 2
    return math.ceil(numerator / (p0 - p1) ** 2)


def evaluate_cohort(converted: int, total: int,
                    expected: float = TIER_A_EXPECTED_RATE) -> dict:
    """Test one completed Tier A cohort against the expected rate.

    Returns the observed rate, a confidence interval, the one-sided p-value,
    and - more useful than the p-value - whether the interval is wide enough
    that the cohort simply cannot answer the question yet.
    """
    observed = converted / total if total else 0.0
    low, high = stats.binomtest(converted, total).proportion_ci(confidence_level=0.95)
    p_value = stats.binomtest(converted, total, expected, alternative="less").pvalue

    if high < TIER_A_ACTIONABLE_FLOOR:
        verdict = "BROKEN"
    elif p_value < ALPHA:
        verdict = "DEGRADED"
    elif low > BASELINE_RATE:
        verdict = "HEALTHY"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n": total, "converted": converted, "observed": observed,
        "ci_low": low, "ci_high": high, "p_value": p_value, "verdict": verdict,
        # An interval spanning both the expectation and the floor means the
        # cohort is too small to distinguish "fine" from "broken" - reporting
        # that honestly is more useful than reporting a non-significant result
        # as though it were reassurance.
        "underpowered": low < TIER_A_ACTIONABLE_FLOOR < high,
    }


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    rule("1. THE CLAIM UNDER TEST")
    print(f"  Tier A is expected to convert at {TIER_A_EXPECTED_RATE:.2%}, "
          f"against a {BASELINE_RATE:.2%} baseline.")
    print(f"  Measured in-sample on 1,200 labelled rows, so it is an upper bound.")
    print(f"  It stops being worth a rep's time below roughly "
          f"{TIER_A_ACTIONABLE_FLOOR:.0%}.")

    rule("2. HOW LONG BEFORE THIS CHECK COULD POSSIBLY NOTICE")
    print(f"  Tier A yields ~{TIER_A_PER_BATCH} accounts per batch, and conversion")
    print(f"  is only observable {CONVERSION_WINDOW_DAYS} days after the call.\n")
    print(f"  {'drop to':<12}{'n needed':<12}{'batches':<11}{'weekly cadence':<18}"
          f"{'monthly cadence'}")
    for target in (0.200, 0.175, 0.150, 0.1333):
        n = required_sample_size(TIER_A_EXPECTED_RATE, target)
        batches = math.ceil(n / TIER_A_PER_BATCH)
        weekly = batches * 7 + CONVERSION_WINDOW_DAYS
        monthly = batches * 30 + CONVERSION_WINDOW_DAYS
        print(f"  {target:<12.1%}{n:<12}{batches:<11}"
              f"{weekly:>5} days ({weekly / 30:.1f} mo){'':<3}"
              f"{monthly:>5} days ({monthly / 30:.1f} mo)")
    print("\n  Read the weekly column. Even running every week with instant data,")
    print("  a fall from 26.7% to 20% takes about five months to establish.")
    print("  That is the two-quarter gap in the Cordilla story, and it is why")
    print("  outcome monitoring alone would have reproduced that failure.")

    rule("3. CALIBRATION — what 'healthy' looks like, on data we have")
    train = pd.read_csv(scoring.REPO_ROOT / "data" / "training_data.csv")
    model, positive_index = scoring.load_model()
    train["p"] = scoring.score_frame(model, positive_index, train)
    train["tier"] = scoring.assign_tier(train.p)
    tier_a = train[train.tier == "A"]
    result = evaluate_cohort(int(tier_a.converted_within_90d.sum()), len(tier_a))
    print(f"  Treating the 1,200 labelled rows as one completed cohort:")
    print(f"    n={result['n']}, converted={result['converted']}, "
          f"rate={result['observed']:.2%}")
    print(f"    95% CI [{result['ci_low']:.2%}, {result['ci_high']:.2%}]  "
          f"verdict={result['verdict']}")
    print("  This is the cohort the expectation was derived from, so agreement")
    print("  is arithmetic rather than evidence. It is here to show the shape of")
    print("  the output, not to validate the model.")

    rule("4. THE CHECK ON HYPOTHETICAL LIVE COHORTS")
    print("  Illustrative, to show where the verdict boundaries fall.\n")
    print(f"  {'scenario':<34}{'n':<6}{'conv':<7}{'rate':<9}{'95% CI':<22}{'verdict'}")
    scenarios = [
        ("one batch, holding up", 28, 7),
        ("one batch, looks bad", 28, 2),
        ("three batches, holding up", 84, 22),
        ("three batches, halved", 84, 11),
        ("ten batches, mild decline", 280, 56),
        ("ten batches, collapsed", 280, 17),
    ]
    for label, n, converted in scenarios:
        r = evaluate_cohort(converted, n)
        ci = f"[{r['ci_low']:.1%}, {r['ci_high']:.1%}]"
        flag = "  (underpowered)" if r["underpowered"] else ""
        print(f"  {label:<34}{n:<6}{converted:<7}{r['observed']:<9.1%}{ci:<22}"
              f"{r['verdict']}{flag}")
    print("\n  The dangerous row is the first one, not the alarming one. A single")
    print("  batch converting at 25% returns HEALTHY - and its interval runs from")
    print("  10.7% to 44.9%, so it is equally consistent with a top tier that has")
    print("  fallen below the actionable floor. 'HEALTHY' on 28 accounts is not")
    print("  reassurance, which is why the underpowered flag is printed beside it")
    print("  rather than left for someone to work out.")
    print("\n  Row two shows the opposite failure: 7.1% is genuinely below")
    print("  expectation and reads DEGRADED, but the interval [0.9%, 23.5%] cannot")
    print("  separate a mild decline from a collapse. Correct direction, useless")
    print("  magnitude. Both rows argue the same thing - pool cohorts before")
    print("  acting on anything except a leading indicator.")

    rule("5. WHAT HAPPENS WHEN IT TRIPS")
    print("  DEGRADED (significant fall, still above the floor)")
    print("     Notify the owning engineer and the sales lead in the same message,")
    print("     since the sales lead will hear it from reps first either way.")
    print("     Cross-reference drift_monitor: if a leading indicator fired weeks")
    print("     earlier, the cause is already known and this is confirmation.")
    print("     Keep shipping the list; tell reps the top tier is running below")
    print("     its usual rate.")
    print()
    print("  BROKEN (upper bound of the interval below the actionable floor)")
    print("     Stop presenting model-ranked tiers. Fall back to the two-fact")
    print("     ordering - vendor data held, trial started - which reached 0.612")
    print("     AUC in-sample at zero cost and needs no model to compute.")
    print("     The fallback was priced during the impact work precisely so this")
    print("     decision does not have to be made under pressure.")
    print()
    print("  INCONCLUSIVE")
    print("     Not a clean bill of health. Keep pooling cohorts and say plainly")
    print("     that the question is still open. The failure mode this whole")
    print("     project is about is an open question being read as a closed one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
