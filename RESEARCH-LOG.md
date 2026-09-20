# Research Log — Cordilla Account Scoring

Written as the work happened, in order. Not reconstructed afterwards.

Entries 1–4 come from a planning session in Claude (chat) before this repo was opened. Entry 5 onward is Claude Code in VS Code, working against the repo directly.

---

## Entry 1 — Planning session — Understanding the problem

**Tool:** Claude (chat).

**Where I started:** before touching any code, I worked through the take-home packet to be sure I understood what Cordilla is actually asking for, what's being graded, and what's explicitly out of scope. I used Claude to pressure-test that reading, under one constraint: only what the document actually says, no invented requirements.

**What that established:**
- The business problem is that Cordilla's reps have no way to prioritise tens of thousands of untouched Prospect / Suspect / Former Customer accounts.
- Three graded dimensions: impact framing, a working agent, and monitoring — with monitoring called out as "the part most take-homes skip."
- A deliverables checklist, and an explicit list of things *not* required: no retraining, no live LLM key, no production pipeline, no formal test suite.
- The scaffold's folder layout.

**What I verified myself:** I read that summary against the packet and the scaffold's `README.md` side by side. The feature list and order — `account_type`, `employee_count`, `industry`, `intent_score`, `mql_count_90d`, `trial_started`, `trial_active_users`, `web_touchpoints_90d`, `sales_contacts_90d` — matches the README exactly, and `account_id` / `snapshot_date` are correctly excluded as identifiers rather than model inputs.

**Terminology I had wrong:** early on I referred to "the newly built LLM" when I meant the provided scikit-learn model. The AI flagged it: `model.pkl` is a plain classifier. An LLM only enters through my own agent design, and it gets mocked. Worth recording because the wrong mental model would have pushed me toward wrapping an LLM around a step that doesn't need one.

## Entry 2 — Planning session — First look at the data

**What I asked:** What the columns mean in practical sales terms, grounded in the actual training file rather than generic assumptions.

**What the data showed** (computed from `training_data.csv`, 1,200 rows):
- Baseline conversion: 78 / 1,200 = **6.5%**. Every lift claim later has to be measured against this.
- `intent_score` is missing on 482 / 1,200 = **40.2%** of rows, matching the packet's note about partial third-party vendor coverage. A real trust caveat, not a cosmetic one — a large share of accounts get scored without that signal at all.
- `account_type` split: 632 Prospect / 402 Suspect / 166 Former Customer. Conversion by type is flat (5.97% / 6.65% / 7.23%) — weak as a standalone driver.
- `trial_started = 1` on 223 / 1,200 rows (18.6%), converting at **9.9%** against the 6.5% baseline. The clearest single lift signal in the raw data.
- Industry is spread roughly evenly across six categories, no obvious skew.

**Where I overrode the AI:** it was ready to discuss "impact" in the abstract before any numbers existed. I refused to accept an impact figure that wasn't computed from `training_data.csv` first. The numbers above are the result of that redirect, and they're the only ones I'll defend.

**Open question carried forward:** I hadn't opened `model.pkl` yet, so I didn't know how the pipeline handles the 40% missing `intent_score` at inference time. That affects both how far to trust the output and what monitoring should watch. Resolve before scoring the live batch.

## Entry 3 — Planning session — Hand-off into Claude Code

**What I asked:** For a full context hand-off (`CLAUDE.md`) so I could continue in Claude Code rather than chat, carrying the assessment context, the data findings above, and how I want to work — incremental, explained, no invented numbers.

**What came back:** `CLAUDE.md` in the repo root, plus this log seeded with the entries above.

**Assumption I'm stating outright:** I'm treating the pre-repo planning conversation as legitimate research-log material. It's the real record of how the problem got broken down before any code existed. The alternative — discarding it and pretending work started fresh in Claude Code — would make the log less honest, not more.

## Entry 4 — Planning session — Checking snapshot freshness before building

**What I asked:** Whether my phase-by-phase roadmap was aligned with the assessment, and to correct anything off.

**What I had computed rather than asserted:** the age of every row's `snapshot_date` against the fixed reference date of 2026-08-01, in both files.

**Result:**
- `training_data.csv`: 1–711 days old, median **280 days**. 73.1% older than 180 days, 32.8% older than a year.
- `accounts_to_score.csv`: 0–675 days old, median **122 days**. 34.0% older than 180 days, 14.0% older than a year.

**Why it matters:** the model was fit mostly on older snapshots than the batch it's about to score. That's a measured train/inference shift on recency — a dimension that plausibly correlates with everything else in the row, since a Suspect untouched for 700 days doesn't behave like one snapshotted last month. It's the kind of gap that leaves aggregate metrics looking fine while calibration quietly slips for the population actually being scored. Feeds both the impact framing (as a stated caveat) and the monitoring design (as a concrete freshness check), and it maps directly onto the packet's planted story about a model that looked right at launch and lost credibility two quarters later.

**Correction to my own plan:** my roadmap had "git commits" and "research log" as late, standalone phases. Both moved to continuous, starting now. Deferring them would have produced exactly the tidied-up final state the packet says not to submit.

**Added to the plan:** an explicit `model.classes_` check during model inspection. Assuming which `predict_proba()` column is the positive class is an easy miss that would silently invert every priority tier downstream.

## Entry 5 — 2026-09-20 — Repo audit and environment setup

**Tool:** Claude Code (Opus 5) in VS Code.

**What I asked first:** To inspect the actual folder contents rather than trusting the structure described in `CLAUDE.md`, and to mark every file as provided / to-build / do-not-modify.

**What that turned up:**
- The company's scaffold is exactly the eight files in commit `12d3ac6`. A clean provided-vs-mine boundary I can point at in the presentation.
- Line counts in that commit independently confirm Entry 2's row counts: 1,201 lines = 1,200 training rows, 301 = 300 rows to score. The planning-session numbers hold against the real files, not just my notes.
- `.gitignore` was excluding `RESEARCH-LOG.md` and `CLAUDE.md`, confirmed with `git check-ignore -v`. `RESEARCH-LOG.md` is a graded deliverable, so that would have dropped it from the submission silently — `git status` says nothing and `git add .` skips it. Caught before the first commit.

**What I asked next:** To set up the environment and confirm the model loads, with the pinned versions untouched.

**The obstacle, and the call I made:** my only interpreter was Anaconda's base Python **3.13.5**. `numpy==1.26.4` (Feb 2024) and `scikit-learn==1.5.2` (Sept 2024) both predate Python 3.13's October 2024 release and ship no 3.13 wheels, so pip falls back to a source build, which fails on Windows without a C/C++ toolchain. The pins are a fixed constraint for this exercise, so the interpreter is what changes, not `requirements.txt`. Created a conda environment on **Python 3.12.14**, the newest version with wheels for all four pins.

**Verified, actually run:**
- `pip list` shows the pins resolved exactly: pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2, matplotlib 3.9.2.
- `pickle.load()` on `model/model.pkl` returns a scikit-learn `Pipeline` with zero warnings captured. That's meaningful: scikit-learn raises `InconsistentVersionWarning` when unpickling an estimator written by a different version, so silence is positive evidence that 1.5.2 is genuinely the version that produced the file. The pipeline internals can be read as-is rather than treated as approximately reconstructed.

**A seam I'm stating rather than papering over:** I did not empirically prove the 3.13 install fails. I predicted it from wheel availability and routed around it instead of spending time reproducing the failure. The claim I'd defend is the positive one — 3.12.14 works, verified end to end.

**Correction to the provided scaffold:** its `README.md` described the supported Python as "3.11+". That reads as permissive but isn't accurate in practice, because 3.13 satisfies it and still breaks the install. I replaced that line with a plain statement of which versions work, and why the dependency pins are what constrain it.

**Deliberately not done yet:** I stopped at `type(model).__name__` and left the pipeline steps, `classes_` ordering, and imputation strategy alone. Environment verification and model inspection are separate questions, and mixing them would mean forming opinions about the model before the step set up to examine it properly.

## Entry 6 — 2026-09-20 — Data inspection, and a hypothesis of mine that didn't survive it

**Tool:** Claude Code (Opus 5). Everything below is reproducible via `analysis/explore_data.py`, which I committed precisely so that every figure in `PROPOSAL.md` has a runnable source rather than sitting in a chat transcript.

**First, re-verification.** I asked for the planning-session numbers to be recomputed from the files rather than carried forward on trust. All of them held exactly: baseline 78/1,200 = **6.50%**, `intent_score` missing 482/1,200 = **40.17%**, conversion by `account_type` 7.23 / 6.65 / 5.97%, trial-starters 22/223 = **9.87%**. Integrity is clean too — no duplicate `account_id`s, no overlap between the two files, no negative values, no categories in the scoring batch that don't appear in training. Worth the few minutes: everything downstream inherits these.

**What changed my mind about which signal matters.** I'd been treating `trial_started` as the headline, per my Entry 2 note. Tested side by side, it isn't:

| Signal | Split | chi2 | p |
|---|---|---|---|
| `intent_score` present vs missing | 8.22% vs 3.94% | 7.98 | **0.0047** |
| `trial_started` 1 vs 0 | 9.87% vs 5.73% | 4.45 | 0.035 |

Whether a third-party vendor covers an account separates converters better than whether the account started a trial.

**The finding I did not expect.** Among the 718 rows that *have* an intent score, the score itself predicts nothing: quartile conversion runs 6.59 / 10.11 / 6.11 / 10.11%, non-monotonic, chi2 p = **0.335**, r = 0.036. So the vendor's decision to cover an account carries signal; the number the vendor sells does not. I want to be careful about how far I push that — it's one synthetic dataset and I'm not claiming it about intent data in general — but within the data I was given, it's the single most interesting thing in the file, and it changes what the model is actually leaning on.

The two facts stack, which gives me real segments instead of one blended average: no-intent/no-trial **3.51%** (n=399), through to has-intent/trial **12.14%** (n=140). A 3.5x spread from two lookups requiring no model at all — which is also the honest bar the model has to clear to be worth anything.

**Now the part I got wrong.** In Entry 4 I flagged the snapshot-age gap (training median 280 days vs scoring batch 121) and argued it mattered because recency "plausibly correlates with everything else in the row," making the model potentially miscalibrated for the population it's about to score. That was a hypothesis stated with more confidence than it had earned, and I asked for it to be attacked rather than confirmed. It doesn't hold:

- Age doesn't predict conversion: quartiles 5.32 / 7.00 / 8.03 / 5.67%, chi2 p = 0.511, r = 0.009.
- Age doesn't correlate with any feature the model uses: all seven Spearman rho between -0.06 and +0.04, every p > 0.15, computed independently in both files.
- Age doesn't explain the intent finding: the gap survives inside every age band (+5.97pp under 180d, +4.85pp at 180-365d, +2.09pp over 365d).
- The segment mix barely moves between the files: 33.2/6.9/48.2/11.7% vs 32.7/6.0/50.7/10.7%.

The measurement was right; the inference drawn from it was not. The shift is real but sits on a dimension that shows no relationship to anything the model consumes or predicts. **This is the correction I'd point to if asked where I overrode AI output** — stronger than my Entry 2 example, because there I pushed back on a process (don't discuss impact before computing it) whereas here a specific, plausible, already-written-down conclusion of mine was falsified by a test I deliberately designed to break it. The alternative was carrying a confident-sounding drift story into the proposal that a panelist could have dismantled in one question.

I'm demoting rather than deleting it: snapshot freshness stays worth watching operationally, because stale snapshots mean reps act on old information whatever the model says. It does not go to the VP as a calibration risk.

**Where the monitoring design actually landed as a result.** The model's strongest input is presence of vendor coverage — something Cordilla doesn't control and which can change with no error, no exception, no failed job. Current coverage is stable — the vendor supplies data on 59.83% of training accounts and 61.33% of the accounts to be scored (missingness of 40.17% and 38.67% respectively, a stable -1.5pp gap either way). If the vendor expanded coverage to 80%, the implicit "missing means low intent" prior would invert for half the population while the scores continued to look entirely reasonable. That is the packet's planted failure story with a mechanism and a measurable number attached, and it's what I intend to build the Phase 11 check around. It's also a caveat that belongs in the impact framing: the ranking is partly driven by who a data vendor sells us records about, and a sharp rep will eventually notice.

**Seams I'm leaving visible:** 78 positives total, so segment cells get thin and I've reported Wilson confidence intervals throughout instead of bare point estimates — the has-intent/trial cell is 17 conversions in 140 rows, CI [7.7, 18.6]. These are associations in one static synthetic extract, not causal claims. And the null result on intent *value* is an absence of evidence at n=718, not proof the vendor's score is worthless.

**Carried into Phase 5, now with a sharp question:** does the pipeline impute `intent_score`, and does it keep a missing-indicator? If it imputes and drops the indicator, it is discarding the strongest signal in the data. If it keeps it, it is leaning on a third-party artifact. Those are different models with different failure modes and different monitoring.

## Entry 7 — 2026-09-20 — Opening the model, and nearly writing it off too early

**Tool:** Claude Code (Opus 5).

**What it is.** `model.pkl` is a `Pipeline`: a `ColumnTransformer` (one-hot on `account_type`/`industry` with `handle_unknown='ignore'`; `SimpleImputer(strategy='median')` on the seven numerics, `remainder='drop'`) feeding a `GradientBoostingClassifier` — 40 trees, depth 2, learning rate 0.05, `subsample=0.7`, `min_samples_leaf=20`. Sixteen features reach the classifier. It's a small, heavily regularised model, not something tuned into the ground.

**The `classes_` check I'd written into the plan.** `classes_ = [0 1]`, so `P(convert) = predict_proba(X)[:, 1]`. The obvious assumption turned out to be right, which is exactly why it was worth confirming — a wrong guess here inverts every priority tier while producing output that still looks entirely plausible. Also established mechanically: the pipeline selects columns **by name** (reversing the column order changes nothing) but **rejects a bare numpy array**. It has to be handed a DataFrame with those exact names.

**The thing I actually went looking for.** The imputer is `median` with `add_indicator=False`, and `statistics_` puts the `intent_score` fill value at **25.3**. Rather than infer the consequence from the config, I tested it:

```
P(convert), intent_score = NaN  : 0.044872
P(convert), intent_score = 25.3 : 0.044872
```

Identical. **The model cannot distinguish "no vendor data" from "the vendor says 25.3."** Entry 6 established that presence of vendor coverage is the strongest single signal in the data (8.22% vs 3.94%, p = 0.0047). This pipeline discards it. The cost is measurable: reality separates those two groups by 4.28pp, the model separates them by 1.78pp, and accounts with no vendor data are over-scored by +1.61pp on average — pushed into call lists despite historically converting at well under half the baseline.

There's a compounding detail. `intent_score` is the model's **top feature at 27.1% importance**, and 40% of its values are a constant frozen at fit time. That lines up uncomfortably well with Entry 6's finding that the *value* of the intent score carries no real signal (p = 0.335): the model has assigned its largest share of importance to a feature that is 40% synthetic and, in the part that isn't, mostly noise.

**Where I nearly went wrong.** At that point the obvious conclusion was "this model is broken, the vendor feature is garbage, recommend against deploying." I didn't write that, because I hadn't yet checked the only question that actually decides it: **does it beat what a rep could do for free?** So I built the comparison before forming the judgement.

| Ranker | AUC (in-sample) |
|---|---|
| The model | **0.759** |
| 4-segment historical rate (has_intent x trial) | 0.612 |
| `has_intent` alone | 0.585 |
| Coin flip | 0.500 |

And in the terms the decision is actually made — work the top N%, reach how many converters:

| Top N% | Called | Model | 2-fact heuristic | Random |
|---|---|---|---|---|
| 5% | 60 | 15 conv, 25.0% hit | 9, 15.0% | 3.9 |
| 10% | 120 | **32 conv, 26.7% hit** | 13, 10.8% | 7.8 |
| 20% | 240 | 42 conv, 17.5% hit | 25, 10.4% | 15.6 |
| 50% | 600 | 61 conv, 10.2% hit | 49, 8.2% | 39.0 |

Working the top 10% yields a 26.7% hit rate against a 6.5% baseline — about **4.1x** — and reaches 41% of all converters from 10% of the calls. It clearly beats the free heuristic. So the honest verdict is not "broken." It's **useful despite a specific, identifiable blind spot**, and its usefulness most likely comes from `web_touchpoints_90d` (21.4%) and `sales_contacts_90d` (20.9%) rather than from the vendor feature it leans on hardest.

**Ranking, not probabilities.** Decile calibration is poor and non-monotonic in the middle — decile 5 converts at 0.83%, below decile 0's 1.67%. Only the top decile is dependable, and there the model predicts 13.9% against an actual 26.7%, under-predicting by nearly half. So the output is fit for ordering a call list and unfit for quoting as a likelihood. On the 300-account batch, scores run 0.038–0.209 with 109 above the 6.5% baseline; the model never claims an account is a sure thing, and I shouldn't let anyone read it that way.

**The seam that matters most.** Every number above is **in-sample** — the model was fit on these 1,200 rows, there's no holdout, and the packet rules out cross-validating. So 0.759 is an upper bound on real skill, not a measurement of it. True out-of-sample performance is simply not knowable from what I was given. That's the single largest unknown in this submission and it goes in the proposal as such rather than being smoothed over. Secondary: the top-decile result is 32 conversions in 120 rows, Wilson interval roughly [19.5%, 35.3%].

**What this changes downstream.** The monitoring design now has two concrete silent-failure mechanisms with a demonstrated causal path to the model's output, which is a real improvement on the snapshot-age idea I killed in Entry 6: (1) if vendor coverage shifts, a different fraction of rows receives the constant 25.3 and the score distribution moves with no error raised; (2) the fill value itself is frozen at 25.3, so if the vendor's own distribution drifts, 40% of rows get systematically wrong values on the model's most important feature.

And a seed for the agent design: I can't touch the model, but the agent can carry `has_intent` as a visible flag beside the score — "this account scores 0.11, but we hold no vendor data on it, and accounts like that convert at 3.94%." That uses the model's ranking without laundering its blind spot into something reps can't see.

## Entry 8 — 2026-09-20 — Scoring the batch and setting tier cutoffs

Scored all 300 accounts via `analysis/score_and_evaluate.py` (300 in, 300 out, asserted in code). Tier counts: A 28, B 46, C 79, D 147. Scores span 0.038–0.209.

**The one judgement call worth recording.** Tiers use **absolute probability thresholds** taken once from the training score distribution (p90 = 0.1088, p75 = 0.0792, p50 = 0.0525), not per-batch percentiles. Percentile tiering would manufacture a full Tier A in every batch regardless of how weak the accounts were, hiding exactly the degradation I want to catch; with fixed cutoffs a poorer batch yields a smaller Tier A and the counts become a free drift signal. This batch produced 9.3% Tier A against the 10% training reference — inside noise, but the mechanism is the point. Validated the cutoffs against actual outcomes before adopting them: 26.67% / 9.29% / 4.04% / 2.83%, cleanly monotonic, where the raw deciles were not.

**A bug I caught in my own design while writing the impact framing.** The first version recomputed the cutoffs with `np.percentile` on every run. That quietly contradicts the whole argument above: a threshold that moves with its input is not something a batch can fall short of, and if the training file were ever swapped the tiers would shift with no visible change. Thresholds are now frozen constants in the module, rounded to 4dp, with a provenance function that re-derives the percentiles and prints the rounding shift (-0.000027 to -0.000049) so the rounding stays auditable. The fix moved three borderline training accounts from C to B and one batch account from D to C. Caught only because two scripts computing the same tiers disagreed by a few rows — a good argument for having the proposal's numbers come from committed code rather than from a scratch buffer. Worth noting that tiers C and D both convert *below* the 6.5% baseline — the useful message to a sales lead is less "here are your best accounts" than "three quarters of this list is worse than random and here is which three quarters."

**A hypothesis I tested and dropped.** I expected Tier A to be contaminated by the over-scored no-vendor-data accounts from Entry 7. It isn't — Tier A has the highest coverage in the batch at 78.6% against a 61.3% average. Still, 6 of the 28 Tier A accounts have no vendor record (rank 7, `ACC-00646`: 637 employees, 6 sales contacts, scored 0.148, no vendor data), so the per-account flag stays in the agent design rather than being dropped as unnecessary.

**Caveat carried forward:** tier lift is measured in-sample on training rows. "Tier A converts at 26.67%" is an upper-bound expectation conditional on the new batch behaving like the old population, which cannot be verified from what I was given. That gap is the reason the monitoring work matters.

## Entry 9 — 2026-09-20 — Framing impact for someone who does not care how the model works

**The reframe that made the rest of this easy.** I started writing this as "what the model improves" and it read like a metrics report. The version I kept starts from the decision instead: an SDR picks which accounts to work this week out of tens of thousands they will never get through, and today that choice is a guess. What the model changes is not that scores exist — it is that the list acquires **an order and a cut line**. That is the entire product. Framing it that way also fixes the counterfactual: the model is not competing against perfect knowledge, it is competing against guessing.

**The unit I chose, and why not the obvious ones.** AUC is meaningless to a VP of Sales, and "4.1x lift" invites the question "4.1 times what?" The unit that survives translation is **calls per conversion**: 3.8 in tier A, 10.8 in tier B, 15.4 calling at random, 35.3 in tier D. *A rep working tier A closes one deal for roughly every four accounts they call; working the list blind, about fifteen.* No ML vocabulary, and every figure is measured rather than asserted.

**I decided to lead with the loss.** Skipping tier D means skipping 50% of the list and giving up 21.8% of the conversions in it. That number argues against my own recommendation, and I am putting it in front of the VP unprompted for two reasons. First it is true. Second, a rep is eventually going to close a tier D account and wave it around as proof the model is wrong — and it lands very differently if the model's own owner said it first. The counter I would give in that conversation is the scarcity point: those accounts were never going to be called anyway. The model is not discarding deals that were otherwise being won, it is allocating attention that is already insufficient.

**Cost of being wrong is asymmetric, and it matters for where the cut line sits.** A false positive — a tier A account that never converts — costs one call, is discovered immediately, and is survivable: 73% of tier A calls do not convert, but under random calling 93.5% do not. The model moves the waste rate, it does not remove it, and I would rather say that plainly than imply it eliminates wasted effort. A false negative — a converter buried in tier D, 17 of 78 in the training data — costs a missed deal, is never observed, and nobody ever learns it happened. Structurally more expensive and structurally invisible. That asymmetry is the argument for keeping tier C reachable instead of deleting it, and it is also the reason the monitoring work is not optional: the expensive failure mode here is the one that produces no signal.

**The batch projection, stated conditionally on purpose.** Twenty-eight tier A accounts, expected to yield roughly 7.5 conversions with a 95% range of 5.5 to 9.9, against 1.8 from calling twenty-eight at random. The "if the batch behaves like the training population" qualifier is load-bearing and I will not bury it — that expectation rests entirely on in-sample performance with no holdout available. It is an upper bound, not a forecast.

**The bar I held it to.** A free two-fact lookup — do we hold vendor data, did they start a trial — reaches AUC 0.612 with no model, no pipeline and nothing to maintain. The model reaches 0.759. It earns its place, but I wanted that comparison written down for two reasons: it is the bar a VP should apply and would not think to ask for, and it means the fallback position is known and costs nothing. If this is ever switched off, reps are not back to guessing, they are back to 0.612.

**What I refused to produce.** The provided data contains no revenue, ACV, deal size, rep headcount, calls per rep per week, cost per call, cycle length, or any record of what reps currently achieve. So no ROI figure, no dollar amount and no payback period appears anywhere in this submission. Given ACV and rep capacity, calls-per-conversion converts to money in a single step — that is the VP's input to supply, and asking for it is a better answer than inventing it. I would rather be asked "why no dollar figure" than be caught having made one up.

**Two caveats I moved out of the footnotes and into the main framing,** because both are things a rep will encounter directly: six of the twenty-eight tier A accounts have no vendor intent data and sit in the group the model over-scores by +1.61pp (named individually in the output, since reps will call them); and the ranking is trustworthy while the probability is not, given the middle deciles are non-monotonic. Order and cut with it. Never quote it as a likelihood.

## Entry 10 — 2026-09-20 — Agent architecture, and the framework question I had to answer twice

**Tool:** Claude Code (Opus 5).

**Three options, weighed on what the data work had already established.** A deterministic batch pipeline with no LLM at all; the same spine with one mocked LLM enrichment step and a guardrail; or an LLM orchestrator with tool-calling that decides which accounts merit investigation. I recommended the middle one.

**Why not the orchestrator, which is the one that matches the job description's vocabulary.** It would replace a threshold comparison I had already validated against real outcomes with a non-deterministic decision nobody can reproduce. The argument I actually care about is not an engineering-taste one: the scenario's whole premise is a scoring system that quietly lost credibility, which means *"why is this account on my list?"* has to have a stable answer. An LLM-routed triage decision cannot be re-derived tomorrow, cannot be diffed when it changes, and cannot be checked against the 26.67% tier A rate I measured. **Auditability is a product requirement here, not a preference**, and that option trades it away for nothing.

**Where the LLM does earn a place, stated without inflation.** It writes the per-account justification and a suggested opening line. It makes no decision — ranking, tiering, flagging and the capacity cut are all upstream of it. Templates handle this badly only because the interesting cases are combinatorial (heavy web engagement with no trial, a dormant trial with heavy sales contact, a large account with no vendor coverage), and the branches multiply. That is a **real but modest** gain, perhaps 80% coverable by a template, which is exactly why `render_fallback_rationale()` is a complete working template rather than a stub. The LLM is an enhancement layer over a functioning baseline, so its marginal value stays measurable and a dead API degrades the output instead of deleting somebody's call list.

**Then the framework question came back.** Context I should be straight about: LangChain/LangGraph/LangSmith came up in an earlier conversation with the interviewer, who then asked informally for "an efficient approach." That is signal about which tools to *consider*, not a reason to adopt them, so I made it clear the decision had to clear the same bar I had just used to reject the orchestrator option.

It does, but not for the reason a job description would suggest, and I want to be precise because this is the part a panel will push on:

- **It is not because the control flow demands it.** One branch, no cycles. Plain function calls would run this correctly. Forward-compatibility ("new nodes later instead of a rewrite") is the *weakest* of the arguments and YAGNI is a fair counter — I would not adopt a framework on that reason alone.
- **It is because the node boundaries become the observability boundaries.** Every node is a span, `run_report.json` is assembled from those spans, and the monitoring work reads that report. With `LANGSMITH_TRACING` set, the identical graph emits the identical spans to a hosted backend with no code change. That is a real mechanism, not a label.
- **And the pipeline stayed exactly as deterministic as the analysis concluded it should be.** Linear/DAG edges, no cycles, no LLM-directed routing. I deliberately did not adopt LangGraph's agentic machinery to look like the framework is being used properly — that would recreate the orchestrator problem I had just argued against.

**Built and verified, not assumed.** Nine nodes, one conditional edge. What I actually checked:

- End to end on the real batch: 300 rows in, tiers A 28 / B 46 / C 79 / D 147 — **identical to `analysis/score_and_evaluate.py`**. Two independent code paths agreeing is the check that would have caught the threshold bug from Entry 8, so it is now a standing one.
- The guardrail's own test suite, 9 cases, **9/9**: accepts clean output on covered and uncovered accounts, and rejects a dropped vendor-data caveat, an invented headcount, "this account will convert", leaked ML vocabulary, an over-long rationale, a caveat attached to a covered account, and a malformed payload. `python -m agent.guardrails`.
- The input gate against four deliberately broken batches: an unseen industry halts (the encoder's `handle_unknown='ignore'` would otherwise silently zero it — a genuinely invisible failure), duplicate account IDs halt, a coverage collapse to 18% halts at −14.8 sigma, and a milder drift to 51.33% **warns and proceeds** at −3.0 sigma.

That last case is the one I would demo. At 51.33% coverage the batch still scores, the warning propagates onto the call sheet, and **Tier A shrinks** rather than staying full — in this specific test batch, from 28 to 21. The exact count depends on which accounts happen to lose coverage, since intent_score is the model's top feature, so I'm not presenting 21 as a fixed constant — but the direction and the mechanism are robust: the frozen thresholds from Entry 8 register a weaker batch instead of manufacturing a full top tier. The design argument from that entry is now observable rather than asserted.

**One deliberate choice about thresholds.** The coverage gate is expressed in standard errors, not fixed percentage points, because the noise in a coverage estimate depends on batch size — a fixed "warn at 5pp" rule would be hair-trigger on 50 accounts and asleep on 5,000. Warn at 3 sigma, halt at 6. On the real batch this reads +0.5 sigma, so normal data correctly reads as normal.

**What the agent refuses to do.** On a halt it writes the diagnosis and produces no call list at all. Handing a rep a subtly wrong list is worse than handing them nothing: nothing prompts a question, whereas a plausible-but-wrong list gets worked and nobody finds out for a quarter. That is the planted failure mode, answered in code rather than in prose.

**Assumption stated rather than hidden:** rep capacity (default 25) is a CLI parameter, not a finding. Headcount and call volume are absent from the provided data.

## Entry 11 — 2026-09-20 — Monitoring, and the check that proved it could not be the alarm

**Tool:** Claude Code (Opus 5).

**The thing I had already built, and why it was not enough.** The agent's input gate inspects one batch and halts on structural damage — an unrecognised industry, duplicate IDs, a coverage collapse at −14.8 sigma. It works and it is tested. It is also structurally incapable of catching the failure this exercise is actually about. Coverage on a 300-account batch carries a ~2.8pp sampling error, so a 3-sigma gate needs roughly 8.4pp of movement before it says anything. A vendor quietly shedding 0.8pp of coverage a run reaches that in about ten weeks, and the gate is silent for every run in between while the model's most important feature degrades underneath it. A threshold on a single observation catches a cliff; the Cordilla story is a slope.

**So the monitoring is explicitly two layers, and the split is the design.** `monitoring/drift_monitor.py` compares runs to each other; `monitoring/outcome_monitor.py` tests the claim against ground truth.

**Why CUSUM rather than a trend line.** CUSUM accumulates deviation from a known target instead of re-testing each observation alone, which is exactly the sensitivity profile needed — responsive to a small persistent shift, unmoved by one-off noise. A regression slope would also find the trend, but needs a window length chosen up front and re-answers from scratch each time; CUSUM carries state, so drift starting today is detected on its own schedule rather than whenever the window happens to align. Textbook parameters, k = 0.5 and h = 5.0, which catches a sustained 1-sigma shift within a few observations at roughly one false alarm per 465 in-control runs.

On the bundled simulation — 18 runs, coverage falling 0.8pp each, about 0.28 sigma per step — **CUSUM flags coverage at run 15 while the single-run check never fires once across all 18**. On two other signals CUSUM leads the point check by one and three runs. The simulation is synthetic and labelled as such in the code; it exists because a detector nobody has watched fire is indistinguishable from one that never fires, and one real run cannot exercise a cross-run check.

**A flaw I caught in my own monitor and fixed rather than shipped.** The first version scaled every signal by its theoretical sampling error. For mean predicted probability on n=300 that is 0.0018, so an entirely unremarkable 0.7% population shift read as 4.1 sigma. That check would page somebody most weeks, get muted within a month, and a muted alarm is precisely how the original Cordilla effort went unwatched — I would have built the failure mode into the thing meant to prevent it. Sampling error is a *floor* on how much a signal moves, not an estimate: real batches differ because territories rotate and campaigns land, not only because of random draw. The monitor now takes the larger of the theoretical floor and the spread actually observed during a burn-in period. Stated limitation, in the code: if drift is already underway during the burn-in, the estimated noise is inflated and the monitor goes quiet — which is the argument for treating any re-baseline as a deliberate, recorded act.

**I also caught the simulator flattering the detector.** My first synthetic history injected less run-to-run noise than sampling error, which is not a thing that happens in production and which quietly made the detector look better than it is. It also meant the burn-in estimator never engaged, so the fix above was never exercised by the demo. Corrected to 3.5pp against the 2.8pp floor.

**Then the lagging check, which produced the most useful finding in this phase — against itself.** `outcome_monitor.py` tests whether Tier A actually converted at the expected 26.67%. Running the power arithmetic rather than assuming it: detecting a fall to 20% needs ~255 Tier A accounts. At ~28 per batch plus the 90-day conversion window, that is **160 days at weekly cadence — about five months. Thirteen months monthly.**

Five months is a couple of quarters. That is the exact interval in the planted backstory where scores stopped matching what reps saw and nobody could say when it started. **An organisation watching only conversion outcomes would reproduce that failure precisely, while doing nothing wrong** — the instrument is simply too slow for the job it looks like it does. That reframes the whole design: the leading indicators are not a nice supplement to outcome tracking, they are the only thing that can move at the speed the failure moves. The outcome check is confirmation, not alarm. I would not have been able to argue that convincingly without computing it.

**A second-order point from the same table.** A single batch of 28 Tier A accounts converting at 25% returns HEALTHY with an interval of [10.7%, 44.9%] — equally consistent with a top tier that has fallen below the level where it earns a rep's time. So the check prints an explicit `underpowered` flag rather than leaving someone to infer it, because "HEALTHY on one batch" read as reassurance is the same error as the original failure, just faster.

**An error I made writing this up.** My first draft of the output narration said a particular row returned INCONCLUSIVE. The code returned DEGRADED. I had written the commentary from what I expected the logic to do instead of reading what it printed — the precise habit this project keeps arguing against. Corrected, and worth recording as the fourth time in this build that checking the output against the claim caught something.

**What happens when something trips** is specified per severity in the code, with an owner and an action, because a check without a defined response is a log line. The escalation terminates somewhere concrete: if the outcome check reads BROKEN, stop presenting model-ranked tiers and fall back to the two-fact ordering — vendor data held, trial started — which reached 0.612 AUC in-sample at zero cost. That fallback was priced during the impact work specifically so the decision would not have to be made under pressure.

---

*Continues below: the written proposal, then a final consolidating entry.*
