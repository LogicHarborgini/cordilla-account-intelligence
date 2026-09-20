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

---

*Continues below as work proceeds: `model.pkl` inspection, the scoring run, impact framing, agent build, monitoring design.*
