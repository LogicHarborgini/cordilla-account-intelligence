"""
Phase 4 — data inspection for the Cordilla account-scoring exercise.

Every number quoted in PROPOSAL.md comes from this script. Run it to reproduce
them. It reads the two provided CSVs and writes nothing.

    python analysis/explore_data.py

Each section states the question it answers, because several of these checks
exist to *falsify* a hypothesis rather than confirm one — which only makes
sense if the question is written down next to the result.
"""

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# The packet fixes "today" at 2026-08-01. Using the system clock here would
# silently change every age figure in the proposal as time passes.
TODAY = pd.Timestamp("2026-08-01")

TRAIN_CSV = "data/training_data.csv"
SCORE_CSV = "data/accounts_to_score.csv"

# Model inputs, in trained order. account_id and snapshot_date are identifiers,
# not features — they are deliberately absent from this list.
FEATURES = [
    "account_type", "employee_count", "industry", "intent_score",
    "mql_count_90d", "trial_started", "trial_active_users",
    "web_touchpoints_90d", "sales_contacts_90d",
]
NUMERIC = [f for f in FEATURES if f not in ("account_type", "industry")]


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def wilson_ci(k, n, z=1.96):
    """95% CI for a proportion.

    Wilson rather than the normal approximation because several segments here
    have fewer than 20 conversions, where the normal interval misbehaves and
    can run below zero.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def rate_line(label, frame, target="converted_within_90d"):
    k, n = int(frame[target].sum()), len(frame)
    lo, hi = wilson_ci(k, n)
    print(f"  {label:<32} n={n:<5} conv={k:<4} rate={k / n * 100:>5.2f}%"
          f"   95% CI [{lo:.1f}, {hi:.1f}]")


def load():
    train = pd.read_csv(TRAIN_CSV)
    score = pd.read_csv(SCORE_CSV)
    for df in (train, score):
        df["age_days"] = (TODAY - pd.to_datetime(df["snapshot_date"])).dt.days
        # Presence of vendor intent data, treated as a signal in its own right.
        df["has_intent"] = df["intent_score"].notna()
    return train, score


def section_shape(train, score):
    rule("1. SHAPE, COLUMNS, MISSINGNESS")
    print(f"training : {train.shape[0]} rows x {train.shape[1]} cols")
    print(f"to_score : {score.shape[0]} rows x {score.shape[1]} cols")
    print(f"\ncolumns present in training but not in scoring batch: "
          f"{set(train.columns) - set(score.columns) - {'age_days', 'has_intent'}}")
    for name, df in (("training", train), ("to_score", score)):
        miss = df[FEATURES].isna().sum()
        miss = miss[miss > 0]
        print(f"\nmissing values — {name}:")
        if miss.empty:
            print("  none")
        for col, n in miss.items():
            print(f"  {col:<20} {n:>5} / {len(df)}  ({n / len(df):.2%})")


def section_baseline(train):
    rule("2. BASELINE CONVERSION — the number every lift claim is measured against")
    rate_line("ALL TRAINING ROWS", train)


def section_signals(train):
    rule("3. THE TWO CANDIDATE SIGNALS")
    print("Question: which single fact best separates converters from non-converters?\n")
    print("intent_score availability:")
    for value in (False, True):
        rate_line(f"has_intent = {value}", train[train.has_intent == value])
    print("\ntrial_started:")
    for value in (0, 1):
        rate_line(f"trial_started = {value}", train[train.trial_started == value])
    print("\naccount_type (included because it is an obvious thing to sort by):")
    for value in sorted(train.account_type.unique()):
        rate_line(value, train[train.account_type == value])

    print("\nsignificance (chi-square on the 2x2):")
    for name, mask in (("has_intent", train.has_intent),
                       ("trial_started", train.trial_started == 1)):
        chi2, p, _, _ = stats.chi2_contingency(
            pd.crosstab(mask, train.converted_within_90d))
        verdict = "significant" if p < 0.05 else "not significant"
        print(f"  {name:<16} chi2={chi2:6.2f}  p={p:.5f}  ({verdict} at 0.05)")


def section_intent_value(train):
    rule("4. IS THE SIGNAL IN *HAVING* AN INTENT SCORE, OR IN ITS VALUE?")
    print("This distinction decides how much the vendor feed is actually worth.\n")
    present = train[train.has_intent].copy()
    present["quartile"] = pd.qcut(present.intent_score, 4,
                                  labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    for q in present.quartile.cat.categories:
        sub = present[present.quartile == q]
        rate_line(f"{q} [{sub.intent_score.min():.1f}-{sub.intent_score.max():.1f}]", sub)

    chi2, p, _, _ = stats.chi2_contingency(
        pd.crosstab(present.quartile, present.converted_within_90d))
    r, p_r = stats.pointbiserialr(present.converted_within_90d, present.intent_score)
    print(f"\n  chi2 across quartiles = {chi2:.2f}, p = {p:.4f}")
    print(f"  point-biserial corr(intent_score, converted) = {r:.4f}, p = {p_r:.4f}")
    print("\n  Read: if p > 0.05 here, the vendor's *coverage decision* carries"
          "\n  signal while the score it sells does not.")


def section_segments(train, score):
    rule("5. THE 2x2 — DO THE SIGNALS STACK?")
    for has in (False, True):
        for trial in (0, 1):
            sub = train[(train.has_intent == has) & (train.trial_started == trial)]
            rate_line(f"has_intent={has}, trial={trial}", sub)

    print("\nsegment mix as % of each file (is the scored population comparable?):")
    for name, df in (("train", train), ("score", score)):
        mix = (df.groupby([df.has_intent, df.trial_started == 1]).size()
               / len(df) * 100).round(1)
        mix.index.names = ["has_intent", "trial"]
        print(f"\n--- {name} ---")
        print(mix.to_string())


def section_age(train, score):
    rule("6. SNAPSHOT AGE — TESTING A HYPOTHESIS, NOT CONFIRMING ONE")
    print("Hypothesis from the planning session: the scoring batch is fresher than")
    print("the training data, and recency 'plausibly correlates with everything")
    print("else in the row', so the model may be quietly miscalibrated.")
    print("Below is the attempt to falsify that.\n")

    for name, df in (("training", train), ("to_score", score)):
        age = df.age_days
        print(f"  {name:<9} min {age.min():>3}  median {age.median():>5.0f}  "
              f"mean {age.mean():>6.1f}  max {age.max():>3}   "
              f">180d {(age > 180).mean():>5.1%}   >365d {(age > 365).mean():>5.1%}")

    print("\n6a. Does age predict conversion at all?")
    train = train.copy()
    train["age_q"] = pd.qcut(train.age_days, 4, labels=["fresh", "Q2", "Q3", "stale"])
    for q in train.age_q.cat.categories:
        sub = train[train.age_q == q]
        rate_line(f"{q} [{sub.age_days.min()}-{sub.age_days.max()}d]", sub)
    chi2, p, _, _ = stats.chi2_contingency(
        pd.crosstab(train.age_q, train.converted_within_90d))
    r, _ = stats.pointbiserialr(train.converted_within_90d, train.age_days)
    print(f"\n  chi2 across age quartiles = {chi2:.2f}, p = {p:.4f}"
          f"   corr(age, converted) = {r:.4f}")

    print("\n6b. Does age correlate with any feature the model actually uses?")
    rows = []
    for col in NUMERIC:
        rho_t, p_t = stats.spearmanr(train.age_days, train[col], nan_policy="omit")
        rho_s, p_s = stats.spearmanr(score.age_days, score[col], nan_policy="omit")
        rows.append({"feature": col, "train_rho": round(rho_t, 4), "train_p": round(p_t, 3),
                     "score_rho": round(rho_s, 4), "score_p": round(p_s, 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n6c. Does the has_intent gap survive inside age bands?")
    print("    (if it vanishes, the signal was an age artifact)")
    bands = pd.cut(train.age_days, [-1, 180, 365, 10_000],
                   labels=["<180d", "180-365d", ">365d"])
    for band in bands.cat.categories:
        sub = train[bands == band]
        rates = {}
        for has in (False, True):
            s = sub[sub.has_intent == has]
            rates[has] = s.converted_within_90d.mean() * 100 if len(s) else float("nan")
        print(f"  {band:<10} no-intent {rates[False]:>5.2f}%   "
              f"has-intent {rates[True]:>5.2f}%   gap {rates[True] - rates[False]:+.2f} pp")


def section_drift(train, score):
    rule("7. TRAIN vs SCORE — DISTRIBUTION COMPARISON")
    cols = NUMERIC + ["age_days"]
    cmp = pd.DataFrame({
        "train_mean": train[cols].mean(), "score_mean": score[cols].mean(),
        "train_median": train[cols].median(), "score_median": score[cols].median(),
    }).round(2)
    cmp["mean_delta_%"] = ((cmp.score_mean - cmp.train_mean)
                           / cmp.train_mean.abs() * 100).round(1)
    print(cmp.to_string())

    print("\ncategorical mix (%):")
    for col in ("account_type", "industry"):
        a = train[col].value_counts(normalize=True).mul(100).round(1)
        b = score[col].value_counts(normalize=True).mul(100).round(1)
        print(f"\n--- {col} ---")
        print(pd.DataFrame({"train_%": a, "score_%": b}).fillna(0).to_string())


def section_integrity(train, score):
    rule("8. INTEGRITY CHECKS")
    print(f"  duplicate account_id      train {train.account_id.duplicated().sum()}"
          f"  |  score {score.account_id.duplicated().sum()}")
    print(f"  account_id overlap between files: "
          f"{len(set(train.account_id) & set(score.account_id))}")
    for name, df in (("train", train), ("score", score)):
        impossible = int(((df.trial_started == 0) & (df.trial_active_users > 0)).sum())
        dormant = int(((df.trial_started == 1) & (df.trial_active_users == 0)).sum())
        started = int((df.trial_started == 1).sum())
        print(f"  {name}: active users without a trial = {impossible} (should be 0);"
              f" trials with zero active users = {dormant}/{started}")
    for col in ("account_type", "industry"):
        unseen = set(score[col].unique()) - set(train[col].unique())
        print(f"  categories in score unseen in train — {col}: {unseen or 'none'}")
    negatives = [c for c in NUMERIC if (train[c] < 0).any() or (score[c] < 0).any()]
    print(f"  features containing negative values: {negatives or 'none'}")


def section_monitoring_baseline(train, score):
    rule("9. THE NUMBER TO MONITOR")
    t = train.intent_score.isna().mean()
    s = score.intent_score.isna().mean()
    print("  The strongest signal found above is whether a third party covers an")
    print("  account. Cordilla does not control that. If coverage shifts, the model's")
    print("  behaviour shifts with it, silently and without raising an error.\n")
    print(f"  intent_score missing — training baseline : {t:.2%}")
    print(f"  intent_score missing — scoring batch     : {s:.2%}")
    print(f"  difference                               : {(s - t) * 100:+.2f} pp")
    print("\n  Established here as the baseline for the Phase 11 monitoring check.")


def main():
    train, score = load()
    section_shape(train, score)
    section_baseline(train)
    section_signals(train)
    section_intent_value(train)
    section_segments(train, score)
    section_age(train, score)
    section_drift(train, score)
    section_integrity(train, score)
    section_monitoring_baseline(train, score)
    print()


if __name__ == "__main__":
    main()
