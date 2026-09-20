"""
Canonical scoring constants and model access. Single source of truth.

WHY THIS FILE EXISTS SEPARATELY: an earlier version of this project defined the
tier cutoffs in two places, and the two copies silently disagreed by three
accounts. Tier boundaries are a policy decision that reaches a sales rep, so
they get exactly one definition and everything else imports it.

Nothing here retrains or mutates the model. It is loaded read-only.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PKL = REPO_ROOT / "model" / "model.pkl"

# The packet fixes "today" at 2026-08-01. Using the system clock would make
# every freshness figure drift as real time passes, which would quietly change
# the monitoring thresholds without anyone editing them.
REFERENCE_DATE = pd.Timestamp("2026-08-01")

# Trained feature order. account_id and snapshot_date are identifiers, not
# inputs — the pipeline drops unknown columns anyway, but passing them would
# misrepresent what the model actually consumes.
FEATURES = [
    "account_type", "employee_count", "industry", "intent_score",
    "mql_count_90d", "trial_started", "trial_active_users",
    "web_touchpoints_90d", "sales_contacts_90d",
]

# Tier cutoffs: FROZEN CONSTANTS, derived once from the p90/p75/p50 of the
# training score distribution and written down, not recomputed per run.
#
# WHY FROZEN: percentile-of-batch tiering would manufacture a full Tier A in
# every batch no matter how weak the accounts were, hiding exactly the
# degradation worth catching. With fixed cutoffs, a weaker batch produces a
# smaller Tier A and the tier counts themselves become a drift signal.
TIER_THRESHOLDS = {"A": 0.1088, "B": 0.0792, "C": 0.0525}

TIER_LABELS = {
    "A": "A - call first",
    "B": "B - work next",
    "C": "C - low priority",
    "D": "D - do not prioritise",
}

# Observed outcomes at those cutoffs, measured on the 1,200 labelled training
# rows by analysis/score_and_evaluate.py. IN-SAMPLE: the model was fit on these
# rows and no holdout exists, so these are an upper bound on real skill, not a
# measurement of it. Used for rep-facing context and monitoring baselines.
TIER_OBSERVED_RATE = {"A": 0.2667, "B": 0.0929, "C": 0.0404, "D": 0.0283}

BASELINE_CONVERSION_RATE = 0.0650          # 78 / 1,200
NO_VENDOR_DATA_CONVERSION_RATE = 0.0394    # 19 / 482
VENDOR_COVERAGE_TRAINING = 0.5983          # 718 / 1,200 have an intent_score


def load_model():
    """Load the pipeline and resolve which predict_proba column means 'converts'.

    WHY RESOLVE IT RATHER THAN HARDCODE [:, 1]: if classes_ were ordered the
    other way, every tier would invert while still producing entirely plausible
    output. Nothing would crash and no test on score ranges would catch it. It
    is verified here once, at load, and raises rather than guessing.
    """
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    classes = list(model.steps[-1][1].classes_)
    if 1 not in classes:
        raise ValueError(f"no positive class 1 in classes_={classes}")
    return model, classes.index(1)


def score_frame(model, positive_index: int, frame: pd.DataFrame) -> np.ndarray:
    """Return P(converts within 90d) for each row.

    The pipeline selects columns by name and rejects a bare numpy array, so a
    DataFrame with the exact feature names must be passed through.
    """
    return model.predict_proba(frame[FEATURES])[:, positive_index]


def assign_tier(scores) -> np.ndarray:
    """Map probabilities to A/B/C/D using the frozen cutoffs above."""
    scores = np.asarray(scores)
    return np.select(
        [scores >= TIER_THRESHOLDS["A"],
         scores >= TIER_THRESHOLDS["B"],
         scores >= TIER_THRESHOLDS["C"]],
        ["A", "B", "C"],
        default="D",
    )


def snapshot_age_days(snapshot_dates) -> pd.Series:
    """Age of each snapshot against the fixed reference date, not the clock."""
    return (REFERENCE_DATE - pd.to_datetime(snapshot_dates)).dt.days
