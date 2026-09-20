# Cordilla Systems Take-Home — Project Context for Claude Code

This file is project memory for Claude Code. Read it fully before writing any code. It carries forward everything established in an earlier planning conversation (Claude, in chat) before the build moved into this repo, so the work in here starts from decisions already made rather than re-deriving them.

---

## 0. Working agreement — carry this forward

- Do NOT dump a full/final implementation in one shot.
- Work in stages. For each stage: (1) explain what we're accomplishing, (2) explain why, (3) give exact files/commands/code, (4) say what output to expect, (5) help interpret the actual output, (6) only then move on.
- If a decision depends on the data, inspect the data first — don't assume.
- Don't invent numbers. Don't assume the model is good or bad without evidence. Don't invent business-impact figures.
- Don't add technology or frameworks to look sophisticated — every choice needs a stated reason tied to an actual requirement.
- Everything in this repo has to be explainable and modifiable live, since the follow-up presentation involves live code changes and a panel extending the scenario on the spot (new constraints, a data source disappearing, a stakeholder objection).
- **Every commit message: no internal phase numbers/labels ("Phase 4," etc.) anywhere in the subject or body.** Describe the actual change and why, as if the reader has never seen this file. Full framing convention (subject/body split, what/why/verification) is in §12 — check it before every commit, not just at session start.

---

## 1. The scenario (condensed from the exercise PDF)

Cordilla Systems: ~900-person B2B workflow software company, sales runs on Salesforce. A few thousand paying accounts, tens of thousands of untouched non-customers (Prospects, Suspects, Former Customers).

VP of Sales's ask (deliberately vague, not a real spec): *"Do something with all this account data so reps stop guessing which accounts to call."*

While scoping this, the in-scenario engineer finds a pre-trained model sitting unused, with no validation history and nobody vouching for it. The job is NOT to re-validate it to research rigor. The job is three things:
1. Decide roughly what this model is worth if put to work.
2. Build something that actually puts its output into an SDR/account manager's real workflow.
3. Design how you'd know, honestly, if it silently stopped being right after you stop watching it.

**Planted detail in the brief:** an earlier Cordilla scoring effort looked great in testing, got real excitement, then quietly lost credibility a couple of quarters post-launch because scores stopped matching what reps saw in the field — and nobody was watching closely enough to catch it while it happened. The PDF calls this "the shape of the job this exercise is testing for." The monitoring design should directly answer that failure mode: silent drift, not a crash.

## 2. What the exercise says it is evaluating

Three things, specifically:
1. Whether the impact is framed in terms a non-engineer (VP of Sales) would care about — not "the model scores accounts."
2. Whether a working agent is actually **built** that does something real with the model's output — not described.
3. Whether there's real thought about how you'd know it's still working after you stop watching it — not just whether it works today.

Code polish beyond what these three need is explicitly NOT the point. Neither is research-grade model auditing.

The PDF explicitly encourages AI-assisted work (Claude Code by name) — what's judged is how the problem gets broken down, how it's prompted, how output is verified, and whether the result can be explained and modified without the tool. Two hard asks: log specific sessions/prompts in `RESEARCH-LOG.md` as you go (not after the fact), and document one specific place where AI output was corrected or overridden.

## 3. Explicit requirements checklist

- [ ] `agent/` — real, working, runs end-to-end. Loads `model/model.pkl`, scores `data/accounts_to_score.csv`, does something real with the output that changes an SDR/AM's day.
- [ ] Agent has defined tools/actions (stated and justified), explicit control flow, a stated framework choice OR a reasoned decision to skip a framework entirely.
- [ ] Any LLM/API call inside the agent is mocked — but with an obvious, documented spot showing: what prompt/instruction it would send, what inputs, what tools it could invoke, what it would return. Judged the same as a real call.
- [ ] `monitoring/` (or folded into `agent/`) — at least ONE concrete, specific, buildable check (health check / data-quality assertion / drift signal / alert condition). Must specify: what signal is watched, what a real problem looks like in that signal vs. normal noise, what happens when it trips. "Add logging" is explicitly disqualifying.
- [ ] `PROPOSAL.md`, 800–1,200 words: impact framing (real numbers, not asserted) + agent design (tools, structure, framework + why, deployment) + monitoring design (what's watched, signal vs. noise, what happens on trip).
- [ ] `RESEARCH-LOG.md`, built live, not retroactively. Hypotheses, what the data said, dead ends, specific AI prompts and what came back. Must include at least one documented correction/override of AI output. Final entry consolidates the numbers/hypotheses/assumptions to defend live (raw material, not a polished presentation).
- [ ] Real, incremental git commit history — not squashed at the end. The PDF says commit history is read as part of the evaluation.
- [ ] `README.md` — setup/run instructions (the scaffold has a skeleton; extend it).
- [ ] `requirements.txt` already pinned — don't change versions: pandas==2.2.3, numpy==1.26.4, scikit-learn==1.5.2, matplotlib==3.9.2.
- [ ] Model feature columns, in trained order: `account_type`, `employee_count`, `industry`, `intent_score`, `mql_count_90d`, `trial_started`, `trial_active_users`, `web_touchpoints_90d`, `sales_contacts_90d`. **`account_id` and `snapshot_date` are identifiers, NOT model inputs — do not feed them to `.predict_proba()`.**
- [ ] Treat **2026-08-01** as "today" for any recency/age calculation — not the system clock.
- [ ] Push to a public git repo, send the link. That is the entire submission — nothing else to package.

## 4. Explicit "do NOT do" list (only things actually stated in the PDF)

- Don't retrain, tune, or improve `model.pkl`.
- Don't audit the model to research-grade statistical rigor or cross-validate it.
- Don't get a real LLM API call working — a well-documented mock is judged identically. No API key is provided.
- Don't build a full deployment pipeline (a rough sketch/config stub is optional, not required — describe the real approach in `PROPOSAL.md` instead).
- Don't try to handle every data-quality edge case — flag only what matters to the impact framing or monitoring design, then move on.
- Don't draw a formal architecture diagram unless it's genuinely faster than writing it out.
- Don't prepare a presentation before submitting — that's a separate follow-up meeting, scheduled after submission.
- Don't modify or regenerate `training_data.csv` or `accounts_to_score.csv`.
- Don't build a formal test suite, packaging, or CI/CD. "It's a working prototype, not a production service."
- Suggested time-box ~4 hours (not a hard cap — the hard cap is a 24-hour submission window from receipt).

## 5. Data already inspected — carry these findings forward, don't re-derive from scratch

Both `data/training_data.csv` (1,200 rows, labeled) and `data/accounts_to_score.csv` (300 rows, unlabeled) share these columns:
`account_id`, `account_type` (Prospect/Suspect/Former Customer), `snapshot_date`, `employee_count`, `industry`, `intent_score` (missing ~40%, mirrors partial third-party vendor coverage), `mql_count_90d`, `trial_started`, `trial_active_users`, `web_touchpoints_90d`, `sales_contacts_90d`, and (training file only) `converted_within_90d` — the target.

**Stats computed and verified from `training_data.csv` (1,200 rows) — use these; don't invent new ones without re-checking:**

- Overall conversion rate: **78 / 1,200 = 6.5%**. This is the baseline any "lift" claim must be measured against.
- `intent_score` missing: **482 / 1,200 = 40.2%** — confirms the PDF's stated ~40% coverage gap. Real for the batch to score too.
- `account_type` distribution: Prospect 632, Suspect 402, Former Customer 166.
- Conversion rate by `account_type`: Suspect 5.97% (24/402), Prospect 6.65% (42/632), Former Customer 7.23% (12/166) — differences are small, not a strong standalone signal.
- `industry` distribution is roughly even across 6 categories (~180–213 rows each: Software, Manufacturing, Retail, Professional Services, Healthcare, Financial Services).
- `trial_started` = 1 for 223/1,200 rows (18.6%). Conversion rate among trial-starters: **22/223 = 9.9%**, vs. 6.5% baseline — the clearest lift signal in the raw data at that point.
- The PDF supplies real-world context to reason about (not measurable from these two CSVs, but it belongs in the impact framing): cold-account outreach converts under 1%; recently-engaged accounts convert low single digits; intent data vendor coverage skews toward larger accounts; product usage telemetry only exists for accounts that started a trial.

**Snapshot-age finding, verified directly from both CSVs — a real, concrete data-quality/trust issue, not an assertion:**

`snapshot_date` age relative to the fixed "today" of 2026-08-01 differs meaningfully between the two files:

- `training_data.csv` (1,200 rows): snapshot ages range from 1 to 711 days old. **Median age 280 days.** 73.1% of rows are older than 180 days; 32.8% are older than a full year.
- `accounts_to_score.csv` (300 rows): snapshot ages range from 0 to 675 days old. **Median age 122 days** — noticeably fresher. Only 34.0% older than 180 days; 14.0% older than a year.

**Why this matters:** the model was fit mostly on fairly old snapshots (median ~9 months), but the live batch it's about to score skews meaningfully fresher (median ~4 months). That's a real train/inference distribution shift on a dimension (recency) that plausibly correlates with everything else in the row — an old Suspect sitting untouched for 700+ days behaves differently than an account snapshotted last week. This is exactly the kind of thing that makes a model *look* fine in aggregate while quietly being miscalibrated for the population it's actually run against, so it feeds both the impact framing (a trust caveat) and the monitoring design (snapshot-age/freshness as a concrete, checkable signal), tying straight back to the "looked good in testing, drifted after launch" story in the brief.

## 6. Repo structure

```
candidate-repo-scaffold/
├── README.md                   setup and run instructions
├── requirements.txt            pinned, provided — do not change versions
├── requirements-agent.txt      optional extras for live LLM providers
├── requirements-notebook.txt   optional, adds Jupyter
├── model/
│   └── model.pkl               provided, don't retrain
├── data/
│   ├── training_data.csv       provided, don't modify
│   └── accounts_to_score.csv   provided, don't modify
├── analysis/                   one-off data and model inspection scripts
├── agent/                      the LangGraph pipeline, scoring, guardrails, providers
├── monitoring/                 drift and outcome checks
├── output/                     generated run artifacts (call sheet, CRM tasks, run report)
├── PROPOSAL.md
├── RESEARCH-LOG.md
└── CLAUDE.md                   this file
```

Model loading pattern confirmed from the scaffold's README (use exactly this):

```python
import pickle
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
# model.predict_proba(df[feature_columns])
feature_columns = [
    "account_type", "employee_count", "industry", "intent_score",
    "mql_count_90d", "trial_started", "trial_active_users",
    "web_touchpoints_90d", "sales_contacts_90d",
]
```

## 7. Where the effort actually pays off

The brief itself flags two places where submissions tend to be thin. Effort goes there, not into extra features:

1. **Monitoring depth.** The PDF calls this "the part most take-homes skip" and ties it to the failure story in §1 (silent drift, not a crash). A concrete, runnable check beats a bullet list of monitoring ideas.
2. **A segmented, defensible impact number**, not a blanket "+X% conversion" claim. Reason by `account_type`, `trial_started`, and `intent_score` availability separately, since those already behave differently in the data, and be explicit about which parts are measurable from the CSVs versus which are informed judgment about the fuller real-world picture (vendor coverage gaps, sub-1% cold outreach) the PDF describes but hands over no data for. A number is more trustworthy with its seams showing than behind false precision.

Two supporting notes:
- A framework, if used, has to be justified by an actual control-flow need — genuine multi-step branching or state — not by name recognition.
- The required "one place you corrected the AI" entry in `RESEARCH-LOG.md` must be a real, specific moment, not a manufactured one.

## 8. Framework choice — LangGraph with tracing around the LLM nodes

The structural default here was a plain Python pipeline, and the control flow genuinely is a straight line. The decision to use LangGraph anyway rests on three independent reasons:

1. **Forward-compatibility tied to a named trigger.** The architecture comparison identified the exact condition that would justify a framework: per-account research where each step depends on the last (CRM notes, email threads, a funding-event search). That's a plausible near-term extension. A `StateGraph` today means that extension is new nodes and conditional edges later, not a rewrite.
2. **Tracing serves observability directly.** LangSmith-style tracing (mocked, with the same documented-seam treatment as the LLM call) is the actual mechanism behind the `run_report.json` observability artifact, and it becomes raw material for the monitoring design.
3. **The pipeline stays exactly as deterministic as the plain-pipeline option was.** Edges form a DAG: `validate_input` → `score_accounts` → `assign_tiers` → `flag_vendor_gap` → `select_call_list` → `generate_rationales` (LLM leaf) → `validate_rationales` (guardrail) → `render_outputs`. The single conditional edge is a deterministic guardrail branch out of `validate_input` to `halt_run` when the input can't be trusted. No cycles, no LLM-directed branching, no non-reproducible decisions. Do not adopt LangGraph's cyclic/agentic machinery just to look like the framework is being fully used — that recreates the exact problem that ruled out the free-roaming-agent option.

`PROPOSAL.md` and `RESEARCH-LOG.md` should present this the same way: the default was no framework, and these are the specific reasons that outweighed it.

## 9. Swappable LLM provider support

The mock path is complete and is what runs by default. On top of it sits a small provider abstraction so the same batch can be run against real models.

**Constraint check first.** §4 says "don't get a real LLM API call working — a mock is judged identically, no key is provided." That is a *floor*, not a *ceiling*: a mock is sufficient, but a real call isn't disallowed. A working, swappable real path added on top of the finished mock adds capability without weakening anything the PDF required.

**Why a provider abstraction rather than one hardcoded vendor.** Wiring a single extra SDK only proves one more integration works. One adapter for any OpenAI-chat-completions-compatible endpoint covers OpenRouter (one key, dozens of models — Claude, GPT, Gemini, Llama, Mistral, DeepSeek), Groq, Together, Fireworks, and a local Ollama server, which is what actually earns the word "various."

- `agent/providers.py` exposes one dispatch function that `agent/mocks.py`'s `call_claude_for_rationale()` calls before falling back to the deterministic mock body.
- The `anthropic` real-call reference code already written out in `agent/mocks.py` is the first working provider (native prompt caching + structured output); the OpenAI-compatible adapter is the second, generalized path.
- Both real paths raise the existing `LLMCallFailed` on any error (bad key, network, malformed response, refusal). `agent/nodes.py`'s existing try/except already catches it and falls back to `render_fallback_rationale()`, so **no node/graph/error-handling change was needed** — the seam absorbed the new capability without touching already-verified code.
- `validate_rationale()` in `agent/guardrails.py` runs on the output regardless of which provider produced it: real model output gets the same fact-checking, no-ML-vocabulary, and caveat-enforcement pass as the mock. Worth stating in `RESEARCH-LOG.md` — the guardrail was designed seam-first, so it needed zero changes to cover real providers.
- `run_report.json` / the `Tracer` records which provider and model produced each rationale (`"mock"` vs. `"anthropic:<model>"` vs. `"openai_compatible:<model>"`), so a live run visibly shows the swap.
- **Dependency discipline:** `anthropic` and `openai` are imported lazily inside the functions that need them, so the default mock path has zero new dependencies and `requirements.txt`'s pinned versions stay untouched. Extra installs are mentioned only in `README.md`, as an optional step.

**Interface — CLI flag for provider choice, env var only for the secret.** Keys never appear on the command line (shell history, `ps`, screen-share risk); they're read from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or a generic `LLM_API_KEY`/`LLM_BASE_URL` pair). Provider *selection* belongs on the command line, because this is a demo-facing feature: `python -m agent.graph --llm anthropic` reads on screen; `export LLM_PROVIDER=anthropic` first does not.

- `--llm mock|anthropic|openai|openai_compatible` (default `mock`, unchanged behavior), `--model <id>` to override the model id within a provider.
- Fail loudly if a live provider is chosen with no matching key in the environment — raise before the run starts, never downgrade to mock silently. A silent downgrade mid-demo is worse than a clean error.
- `--compare provider1,provider2` runs the same batch through multiple providers and prints **guardrail pass rate per provider** side by side. Because `validate_rationale` is provider-agnostic, this turns "add a key" into a real model-comparison eval harness: same 25 accounts, two providers, how many outputs from each pass the same invented-numbers / likelihood-language / dropped-caveat checks.

**In `RESEARCH-LOG.md`:** log this as its own entry, with the constraint check above stated explicitly. Be honest about what was actually tested — only providers a real key was available for can be marked verified; the rest are written against documented APIs and marked untested. Do not claim a live run that didn't happen.

## 10. Status

Build is complete end to end; the repo has 14 incremental commits from scaffolding through the pluggable rationale backend.

- [x] Data inspection, both CSVs — stats in §5 computed and verified.
- [x] `model.pkl` inspected: pipeline steps, preprocessing, `model.classes_` ordering confirmed, median imputation of `intent_score` documented as discarding the strongest available signal.
- [x] All 300 accounts scored; fixed tier cutoffs frozen as constants.
- [x] Impact framing recorded, with its unit stated and the wider-population caveat added.
- [x] Agent built (`agent/`), refuses to run on input it cannot trust.
- [x] Monitoring built (`monitoring/`) for the slow-decline failure mode.
- [x] `PROPOSAL.md` written; `RESEARCH-LOG.md` kept throughout with a consolidating final entry.
- [x] Pluggable LLM provider backend (§9), plus the guardrail bug it exposed, fixed.
- [ ] Final self-review pass and push to the public repo.

## 11. Working roadmap and the rules behind it

The phase-by-phase roadmap below was drafted up front and then corrected in three places before being adopted. The corrections matter more than the list:

**Correction 1 — git commits are NOT a late phase.** The original draft listed "git history" near the end. Wrong: commit after *every* phase, from the first one, not saved up for a dedicated late step.

**Correction 2 — `RESEARCH-LOG.md` entries are NOT a late phase either.** Same fix: log a real entry after each phase as it happens (hypothesis → what the data/model actually showed → decision → any AI back-and-forth), not batched at the end.

**Correction 3 — two concrete additions:**
- Model inspection must explicitly check `model.classes_` before trusting which `predict_proba()` column is the positive ("will convert") class — a common, easy-to-miss bug.
- Data understanding must explicitly quantify `snapshot_date` age/freshness in both files, not skim it — see §5. That finding carries forward into monitoring as a concrete freshness/drift check.

**Pacing note:** given the ~4 hour soft time-box against a 24-hour hard deadline, move quickly through the low-judgment phases (setup, bookkeeping, scoring mechanics) and spend the real back-and-forth where judgment differentiates the work: data, model internals, impact framing, agent design, monitoring. The architecture comparison should be a few sentences per option, not a design doc — the assessment says code polish beyond what's needed isn't the point.

The phase list, each ending with a real commit + a real research-log entry before moving on:

1. Inspect the actual provided folder (not an assumed structure): every file/dir, what each is for, marked provided vs. to-build vs. do-not-modify.
2. Environment setup on Windows/VS Code/Python/Git: venv, activate, `pip install -r requirements.txt`, verify Python/Git versions, verify the model loads, with expected output for each command.
3. Full inspection of `training_data.csv` then `accounts_to_score.csv`: shape, dtypes, missingness, conversion rate overall and by segment, trial vs. non-trial, plus the snapshot-age finding. Plain-language explanation for every number, no invented conclusions.
4. Inspect `model.pkl`: pipeline type, preprocessing steps, expected feature columns/order, input shape (DataFrame vs. array), what `.predict_proba()` returns, **`model.classes_` ordering**, whether missing values are handled internally; plus a throwaway script that loads it, scores a few rows, prints the result.
5. Minimal correct scoring pipeline for all 300 accounts, probability added as a column, then score-distribution analysis (min/max/mean/median/percentiles, top 10/20/50% tiers, account-type mix per tier) — all from actual output, no assumed shape.
6. Impact framing: who decides, what decision changes, cost of being wrong in each direction, numbers computed from the data above. Anything not calculable from the provided data (revenue, rep count, deal size, ROI) is explicitly labeled unavailable or an assumption, never invented.
7. Propose 2–3 lightweight agent architecture options (what it does, inputs/outputs, tools, control flow, LLM needed or not, framework justified or not, tradeoffs), then recommend one for this exercise specifically — not the most impressive-sounding one.
8. Build the recommended agent incrementally. Any LLM call is a clearly documented mock with an explicit "real API plugs in here" seam (prompt, inputs, tools, expected return shape).
9. Structured output schema per account, fields chosen to fit the workflow decided in step 7 and each one justified — not an example schema copied blindly.
10. Monitoring aimed at the *silent* failure mode from §1: a small number of genuinely useful checks (data quality/freshness including the snapshot-age gap, score-distribution drift, and if feasible a lagging business-outcome check). For each: the baseline, what's normal vs. a real problem, what happens when it trips. At least one concrete enough to run now.
11. Write `PROPOSAL.md` (800–1,200 words, the three sections, numbers only from what was actually established above).
12. *(continuous — see Correction 2)* Finalize `RESEARCH-LOG.md` with a consolidating last entry pulling together every number, hypothesis and assumption to defend live.
13. *(continuous — see Correction 1)* Real incremental commits throughout, one per step at minimum.
14. Final self-review: a requirement-by-requirement checklist (Requirement / Implemented? / Where? / Evidence / Missing?), plus an honest list of gaps, weak assumptions, unnecessary complexity, and likely panel questions.

**Rules to keep enforcing throughout:** don't jump ahead of the current step; never invent numbers — label assumptions explicitly; never retrain or modify `model.pkl`; don't over-engineer past prototype level; don't add frameworks without a real justification; explain the why behind every decision, including alternatives considered; treat the exercise PDF as source of truth and flag what it leaves unspecified rather than silently deciding; inspect actual files, data and output before drawing any conclusion; stop at the end of each step and wait for input before continuing.

## 12. Logging, commit and writing conventions — apply automatically

**`RESEARCH-LOG.md` pacing.** Not every step deserves the same depth. Write a full narrative entry (hypothesis → what was actually observed → decision → any AI back-and-forth or correction) for the steps where real judgment gets exercised — data, model internals, impact framing, agent design, monitoring — and for any moment where something unexpected happened (a blocker, a bug, a wrong assumption caught). For mechanical steps, 3–5 lines is enough: what was done, what the output was. Padding routine steps to the same depth wastes time against the box and dilutes the entries that matter.

**Commit messages.** Describe the actual work and why, for a reader with no access to this file. Never reference internal phase numbers ("Phase 3 setup") — a panel reading `git log` has no idea what that means, so phase labels belong in `RESEARCH-LOG.md`, not commits.

Frame every commit as two parts, using `git commit -m "<subject>" -m "<body>"`:

- **Subject** — imperative mood, the concrete outcome, not the activity. "Set up verified Python 3.12 environment," not "Working on environment." Readable on its own in `git log --oneline`.
- **Body** — one short paragraph covering, in order: (1) *what* changed, specifically (files/behavior, not vague summaries), (2) *why* — the actual reason or problem it solves, especially if it wasn't the obvious first approach, (3) *verification* — what was actually run or checked, not just that code was written. If nothing meaningful needs saying beyond the subject, skip the body rather than padding it.

Bad: `"progress"`, `"fix stuff"`, `"updates"`. Good: `"Score all 300 accounts and add priority tiers"` with a body noting the tier cutoffs, why those cutoffs, and that the output row count was checked against the input (300 in, 300 out). Each commit should be small enough that its diff and message obviously match — if the message needs three unrelated sentences, it's two commits.

**Before treating any entry as final:** if a commit message or log entry cites a specific artifact (a commit hash, a line count, a version number), verify it against the real repo state first — never let a placeholder or assumed value ship as if it were checked.

**Writing the deliverables.** `PROPOSAL.md`, `README.md`, `RESEARCH-LOG.md` and code comments all get read by people outside engineering as well as inside it, so:

- Mirror the exercise's own three named dimensions (impact framing / agent / monitoring) in the headers, so the mapping is unambiguous.
- Every claim states its number and its source — which script, which file. The strength of this submission is that everything reproduces on demand; traceable beats impressive-sounding.
- No ML vocabulary in headline claims. Lead with the decision and the tradeoff, not the model.
- Name the vendor-coverage blind spot precisely: it's an **information asymmetry**, not just a data-quality gap. Cordilla's ranking depends on a third party's undisclosed, unmonitored coverage decisions. That's a sharper and more accurate framing of the same risk.

This is about emphasis and framing, not substance or numbers. The work is verified at every step, so writing it clearly and writing it honestly are the same task here.
