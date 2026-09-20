"""
Scores the 300-account batch and evaluates how much the model is worth.

    python analysis/score_and_evaluate.py

Two jobs in one script, deliberately:

  1. Evaluate. Every performance figure quoted in PROPOSAL.md is produced here,
     so a reader can re-run it rather than take it on trust.
  2. Score. Produces output/scored_accounts.csv, the artifact the agent consumes.

A note on tier cutoffs, since this is the main judgement call in this file.
They are ABSOLUTE probability thresholds derived once from the training score
distribution, not per-batch percentiles. Percentile tiering would guarantee a
full Tier A in every batch no matter how weak the accounts actually were, which
hides exactly the degradation worth catching. With fixed thresholds, a batch of
poor accounts produces fewer Tier A rows, and the tier counts become a drift
signal at no extra cost.

Nothing here retrains or modifies the model. It is loaded and called only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import pickle

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

MODEL_PKL = "model/model.pkl"
TRAIN_CSV = "data/training_data.csv"
SCORE_CSV = "data/accounts_to_score.csv"
OUT_CSV = "output/scored_accounts.csv"

# Trained feature order. account_id and snapshot_date are identifiers and are
# deliberately excluded — the pipeline drops unknown columns anyway, but passing
# them would misrepresent what the model consumes.
FEATURES = [
    "account_type", "employee_count", "industry", "intent_score",
    "mql_count_90d", "trial_started", "trial_active_users",
    "web_touchpoints_90d", "sales_contacts_90d",
]

# Tier cutoffs. FROZEN CONSTANTS, not recomputed per run.
#
# Derived once from the p90/p75/p50 of the training score distribution and then
# written down here, rounded to 4dp. Recomputing them at runtime would defeat
# the point: thresholds that move with their input are not thresholds a batch
# can fall short of, and the tier counts would stop being a drift signal. In a
# real deployment these would live in config and change only by deliberate
# review. print_threshold_provenance() re-derives them so the rounding stays
# auditable.
TIER_THRESHOLDS = {"A": 0.1088, "B": 0.0792, "C": 0.0525}
TIER_PERCENTILES = {"A": 90, "B": 75, "C": 50}
TIER_LABELS = {
    "A": "A - call first",
    "B": "B - work next",
    "C": "C - low priority",
    "D": "D - do not prioritise",
}


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_model():
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    classes = list(model.steps[-1][1].classes_)
    # Never assume column 1 is the positive class; a silent inversion here would
    # reverse every tier while still producing plausible-looking output.
    if 1 not in classes:
        raise ValueError(f"no positive class 1 in classes_={classes}")
    return model, classes.index(1)


def score(model, pos_idx, df):
    return model.predict_proba(df[FEATURES])[:, pos_idx]


def print_threshold_provenance(train_scores):
    """Show where the frozen constants came from, and what rounding cost."""
    print("  frozen constant vs the percentile it was derived from:")
    for tier, pct in TIER_PERCENTILES.items():
        exact = float(np.percentile(train_scores, pct))
        frozen = TIER_THRESHOLDS[tier]
        print(f"    Tier {tier}: using {frozen:.4f}   "
              f"(training p{pct} = {exact:.6f}, rounding shift {frozen - exact:+.6f})")
    print("  Rounding moves a handful of borderline accounts. That is accepted:")
    print("  a published, stable cutoff is worth more than 6dp of precision.")


def assign_tier(scores, thresholds=None):
    thresholds = thresholds or TIER_THRESHOLDS
    return pd.Series(
        np.select(
            [scores >= thresholds["A"], scores >= thresholds["B"], scores >= thresholds["C"]],
            ["A", "B", "C"],
            default="D",
        ),
        index=getattr(scores, "index", None),
    )


def section_evaluate(train, y):
    rule("1. IS THE MODEL WORTH USING? (in-sample — favours the model)")
    print("  The model was fit on these rows and no holdout exists, so every")
    print("  number here is an UPPER BOUND on real skill, not a measurement.\n")

    has_intent = train.intent_score.notna()
    trial = train.trial_started == 1
    seg_rate = train.groupby([has_intent, trial]).converted_within_90d.mean()
    heuristic = np.array([seg_rate.loc[(h, t)] for h, t in zip(has_intent, trial)])

    print(f"  model AUC                                  : {roc_auc_score(y, train.p):.4f}")
    print(f"  4-segment historical rate (no model at all): {roc_auc_score(y, heuristic):.4f}")
    print(f"  'do we have vendor data?' alone            : "
          f"{roc_auc_score(y, has_intent.astype(int)):.4f}")
    print("  coin flip                                  : 0.5000")

    rule("2. WHAT A REP ACTUALLY EXPERIENCES: work the top N%")
    n_total, n_conv = len(train), int(y.sum())
    print(f"  {n_conv} converters among {n_total} accounts (baseline {n_conv / n_total:.2%})\n")
    print(f"  {'top N%':<9}{'called':<9}{'converters reached':<22}{'hit rate':<12}"
          f"{'lift vs baseline'}")
    for pct in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = int(round(n_total * pct))
        idx = np.argsort(-train.p.values)[:k]
        hits = int(y[idx].sum())
        hit_rate = hits / k
        print(f"  {pct:<9.0%}{k:<9}{hits:>3} of {n_conv} ({hits / n_conv:>5.1%})      "
              f"{hit_rate:>7.2%}     {hit_rate / (n_conv / n_total):>5.2f}x")
    return heuristic


def section_calibration(train):
    rule("3. CALIBRATION — is the number trustworthy, or only the ordering?")
    train = train.copy()
    train["decile"] = pd.qcut(train.p, 10, labels=False, duplicates="drop")
    cal = train.groupby("decile").agg(
        n=("p", "size"), predicted=("p", "mean"), actual=("converted_within_90d", "mean"))
    cal["predicted"] = (cal.predicted * 100).round(2)
    cal["actual"] = (cal.actual * 100).round(2)
    cal["error_pp"] = (cal.predicted - cal.actual).round(2)
    print(cal.to_string())
    print("\n  Read: if the middle deciles are non-monotonic, the model ranks better")
    print("  than it calibrates, and the probability should not be quoted as a")
    print("  literal likelihood to anyone.")


def section_blind_spot(train):
    rule("4. THE KNOWN BLIND SPOT: accounts with no vendor data")
    train = train.copy()
    train["has_intent"] = train.intent_score.notna()
    g = train.groupby("has_intent").agg(
        n=("p", "size"), model_says=("p", "mean"), reality=("converted_within_90d", "mean"))
    g["model_says"] = (g.model_says * 100).round(2)
    g["reality"] = (g.reality * 100).round(2)
    g["over_score_pp"] = (g.model_says - g.reality).round(2)
    print(g.to_string())
    print("\n  The pipeline imputes a missing intent_score to the training median")
    print("  (25.3) with add_indicator=False, so the model cannot tell 'no vendor")
    print("  data' from 'vendor says 25.3'. Uncovered accounts are over-scored.")
    print("  The agent surfaces this per-account rather than hiding it.")


def section_thresholds(train, y):
    rule("5. TIER CUTOFFS (frozen constants)")
    for tier, thr in TIER_THRESHOLDS.items():
        print(f"  Tier {tier}: score >= {thr:.4f}")
    print(f"  Tier D: everything below {TIER_THRESHOLDS['C']:.4f}\n")
    print_threshold_provenance(train.p.values)

    print("\n  Lift actually observed at these cutoffs in the training data:")
    tiers = assign_tier(train.p)
    base = y.mean()
    summary = pd.DataFrame({"tier": tiers, "converted": y}).groupby("tier").agg(
        n=("converted", "size"), converted=("converted", "sum"), rate=("converted", "mean"))
    summary["rate_%"] = (summary.rate * 100).round(2)
    summary["lift"] = (summary.rate / base).round(2)
    print(summary[["n", "converted", "rate_%", "lift"]].to_string())


def section_batch(scored):
    rule("6. THE LIVE BATCH: 300 ACCOUNTS SCORED")
    print(scored.p.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).round(4).to_string())
    print(f"\n  The model's ceiling on this batch is {scored.p.max():.4f}. It never")
    print("  claims an account is a sure thing; the strongest available claim is")
    print("  'several times more likely than average'.")

    rule("7. TIER COUNTS ON THE LIVE BATCH")
    counts = scored.tier.value_counts().reindex(["A", "B", "C", "D"]).fillna(0).astype(int)
    for tier, n in counts.items():
        print(f"  {TIER_LABELS[tier]:<24} {n:>4}  ({n / len(scored):>5.1%})")
    print("\n  These counts are meaningful because the thresholds are fixed. A batch")
    print("  of weaker accounts would produce a smaller Tier A rather than")
    print("  silently relabelling its best 10% as 'call first'.")

    rule("8. WHO IS IN EACH TIER?")
    print("account_type mix by tier (row %):")
    print((pd.crosstab(scored.tier, scored.account_type, normalize="index") * 100)
          .round(1).to_string())
    print("\nvendor data coverage by tier:")
    cov = scored.groupby("tier").agg(
        n=("p", "size"),
        has_vendor_data=("has_intent", "sum"),
        coverage_pct=("has_intent", lambda s: round(s.mean() * 100, 1)),
        mean_score=("p", lambda s: round(s.mean(), 4)))
    print(cov.to_string())
    print("\n  Watch the coverage column. If Tier A were dominated by accounts with")
    print("  no vendor data, the model's blind spot would be concentrated exactly")
    print("  where reps spend their time.")

    print("\ntrial activity by tier:")
    print(scored.groupby("tier").agg(
        trials_started=("trial_started", "sum"),
        mean_web_touchpoints=("web_touchpoints_90d", lambda s: round(s.mean(), 2)),
        mean_sales_contacts=("sales_contacts_90d", lambda s: round(s.mean(), 2)),
    ).to_string())


def main():
    model, pos_idx = load_model()
    train = pd.read_csv(TRAIN_CSV)
    batch = pd.read_csv(SCORE_CSV)

    train["p"] = score(model, pos_idx, train)
    y = train.converted_within_90d.values

    print(f"positive class index in classes_ = {pos_idx} "
          f"-> P(convert) = predict_proba(X)[:, {pos_idx}]")

    section_evaluate(train, y)
    section_calibration(train)
    section_blind_spot(train)

    section_thresholds(train, y)

    batch["p"] = score(model, pos_idx, batch)
    batch["has_intent"] = batch.intent_score.notna()
    batch["tier"] = assign_tier(batch.p).values
    batch = batch.sort_values("p", ascending=False).reset_index(drop=True)
    batch["rank"] = np.arange(1, len(batch) + 1)

    section_batch(batch)

    # Row-count check: a silent row drop during scoring would be invisible otherwise.
    assert len(batch) == len(pd.read_csv(SCORE_CSV)), "row count changed during scoring"

    Path("output").mkdir(exist_ok=True)
    cols = ["rank", "account_id", "tier", "p", "has_intent", "account_type",
            "industry", "employee_count", "intent_score", "trial_started",
            "trial_active_users", "mql_count_90d", "web_touchpoints_90d",
            "sales_contacts_90d", "snapshot_date"]
    batch[cols].to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(batch)} scored accounts -> {OUT_CSV}")


if __name__ == "__main__":
    main()
