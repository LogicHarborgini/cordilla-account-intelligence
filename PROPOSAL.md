# Cordilla account scoring — proposal

Every number below is reproducible via `analysis/explore_data.py` and
`analysis/score_and_evaluate.py`. Figures from the labelled training data are **in-sample** —
no holdout exists — so they are an upper bound.

---

## 1. Impact framing

**The decision that changes.** A rep picks which accounts to work this week from tens of
thousands they will never reach. Today that is a guess. What changes is not that scores
exist — the list acquires **an order and a cut line**. That is the product.

**Calls per conversion is the unit:** about **four** per deal in the top tier, **fifteen**
working blind.

| Working | Converts at | Calls per conversion |
|---|---|---|
| Tier A | 26.67% | **3.8** |
| Tier B | 9.29% | 10.8 |
| At random | 6.50% | 15.4 |
| Tier C | 4.04% | 24.7 |
| Tier D | 2.83% | 35.3 |

**What the cut costs.** Skipping Tier D skips half the list and gives up **21.8%** of its
conversions — worth saying unprompted, since a rep will eventually close one and call it
proof the system is wrong. Those accounts were not being called anyway.

**Who these numbers describe.** The labelled cohort is not the untouched population the VP is
pointing at: 59.4% have already been contacted by sales, 51.4% are already MQLs, 18.6%
started a trial, and only **5.1% are cold on every signal**. The brief puts cold outreach well
under 1%; this cohort converts at 6.50%. So the tier rates hold for accounts resembling that
slice, and should not be extrapolated to the untouched majority.

**The honest bar.** A free two-fact lookup — vendor data held, trial started — reaches 0.612
AUC with nothing to maintain. The model reaches **0.759**. It earns its place, and the
fallback if switched off is known and free.

**The risk worth naming: an information asymmetry, not a data-quality gap.** 40.17% of
accounts have no third-party intent data, and that fact is the strongest predictor in the
dataset — covered accounts convert at 8.22% against 3.94% (p = 0.0047) — while the *value* the
vendor sells carries no signal at all (p = 0.335). So Cordilla's ranking is substantially
driven by **which accounts a third party chose to cover**, under a policy Cordilla cannot
observe, verify, or be told about when it changes. The brief's related claim that coverage
skews to larger accounts is directionally true but weak here: 126.5 vs 113.2 employees,
Welch's p = 0.2545.

The model compounds this: a missing intent score is imputed to 25.3 and cannot be told from a
real one, so uncovered accounts are over-scored by **1.61pp** against their actual 3.94%. Six
of the 28 Tier A accounts are in that group, flagged individually rather than absorbed into
the ranking.

**What I will not supply.** The data holds no revenue, deal size, headcount or call capacity,
so no ROI figure or payback period appears anywhere here. Given ACV and capacity,
calls-per-conversion converts to money in one step — that input is yours.

---

## 2. Agent design

**What it produces.** `python -m agent.graph` turns the batch into a **call sheet** (ordered,
cut to capacity, one brief per account), **Salesforce Task payloads** — the action attached,
mocked at the boundary — and a **run report** with data-quality findings, tier counts and trace.

**Structure.** A LangGraph `StateGraph` with linear/DAG edges and one conditional branch
(`--print-graph` emits it as mermaid):

```
validate_input ─┬─(proceed)→ score → tier → flag_vendor_gap → select_to_capacity
                │            → generate_rationale → validate_rationale → render
                └─(halt)───→ halt_run
```

**Why a framework, honestly.** Not because control flow demands it: one branch, no cycles;
plain functions would run this correctly. What holds is that **node boundaries
become observability boundaries**: every node is a span, the run report is assembled from
them, and the monitoring reads that report. `LANGSMITH_TRACING` ships the node spans to a
hosted backend unchanged; the LLM and guardrail sub-spans are my own `Tracer`, one
`@traceable` each. Local by default is deliberate: `run_report.json` reproduces from a clean
clone. I did *not* adopt its
cyclic or LLM-routing machinery: every decision reaching a rep stays deterministic, because in
a scenario whose premise is lost credibility, *"why is this account on my list?"* must have a
stable answer.

**Where the LLM sits.** At a leaf, writing the justification for a list that is already
final; it cannot promote, demote or exclude anything. The gain over templates is real but
modest, which is why the template fallback is a complete implementation rather than a stub:
a dead API degrades output instead of deleting somebody's call list.

**Guardrails, because shape is not truth.** The schema guarantees the response parses; it
says nothing about whether a fact was invented. `validate_rationale`
rejects invented numbers, likelihood language ("this account will convert"), leaked internal
vocabulary, and — most importantly — a dropped vendor-data caveat. Its own suite passes
**9/9** (`python -m agent.guardrails`), covering each failure in both directions.

**The gate is the product decision.** On an unrecognised industry, duplicate IDs, or a
coverage shift beyond sampling noise, the agent **writes the diagnosis and produces no call
list**. A rep handed nothing asks why; a rep handed a subtly wrong list works it, and
nobody finds out for a quarter.

**Deployment.** A scheduled Cloud Run container — artifacts to GCS, tasks to Salesforce,
run reports to LangSmith, monitors as a separate job.

---

## 3. Monitoring design

**Two layers, because a threshold on one batch catches a cliff and this failure is a slope.**
Coverage on 300 accounts carries a 2.8pp standard error, so a 3-sigma gate needs ~8.4pp of
movement. A vendor shedding 0.8pp per run reaches that in ten weeks — silent throughout.

**Leading indicators** (`monitoring/drift_monitor.py`) compare runs to each other using CUSUM,
which accumulates small deviations instead of testing each alone. Watched: vendor coverage
(target 59.83%), Tier A share (10% by construction — free drift detection from using frozen
cutoffs rather than per-batch percentiles), mean score, and guardrail rejection rate. On an
18-run simulation with coverage falling 0.8pp per run, **CUSUM flags it at run 15 while the
single-run check never fires once.** Noise comes from a burn-in, not sampling error, which is
only a *floor* on real batch variation.

**The lagging check** (`monitoring/outcome_monitor.py`) tests the actual claim: did Tier A
convert at ~26.7%? It also shows why it cannot be the alarm. Detecting a fall to 20%
needs ~255 Tier A accounts; at ~28 per batch plus the 90-day conversion window that is
**about five months even at weekly cadence**. That is the two-quarter gap in Cordilla's own
history: a team watching only conversions would reproduce that failure while doing nothing
wrong. It is confirmation, not detection.

**What happens when something trips.** *Halt* — the run already stopped; page the engineer;
reps use existing prioritisation for one day; do not re-run until the input is explained.
*Sustained drift* — runs continue, the call sheet carries a data-quality banner so reps see it
where they work; ticket within a day, and any re-baseline is deliberate and recorded, because
doing it silently is how the last effort lost credibility. *Outcome broken* —
stop presenting model-ranked tiers and fall back to the two-fact ordering, which was priced
at 0.612 AUC precisely so that decision need not be made under pressure.
