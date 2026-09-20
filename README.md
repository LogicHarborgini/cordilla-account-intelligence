# Cordilla Systems, AI Engineer Exercise — Impact Framing, Agent Build, Monitoring

## Setup

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

Tested against Python 3.11+ with the exact pinned versions above. If you'd rather work in a notebook than plain scripts (either is fine, see the take-home packet), `pip install -r requirements-notebook.txt` instead (adds Jupyter on top of the same pinned core).

**Use Python 3.11 or 3.12. Python 3.13 is not supported for this project.**

The assessment pins `numpy==1.26.4` and `scikit-learn==1.5.2`, which are not compatible with Python 3.13 through the required pre-built wheels. Using Python 3.11 or 3.12 allows the pinned dependencies to install normally.

The environment has been verified with **Python 3.12.14**, including successful loading of `model/model.pkl` with no version warnings.

If using Conda:

```bash
conda create -n cordilla python=3.12 -y
conda activate cordilla
pip install -r requirements.txt
```

Loading the model (already trained, don't retrain it):

    import pickle
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    # model.predict_proba(df[feature_columns]), feature columns are listed below and in the take-home packet

Expected feature columns, in the order the model was trained on: `account_type`, `employee_count`, `industry`, `intent_score`, `mql_count_90d`, `trial_started`, `trial_active_users`, `web_touchpoints_90d`, `sales_contacts_90d`. `snapshot_date` and `account_id` are identifiers, not model inputs.

**Treat 2026-08-01 as "today" for this exercise.** Both CSVs are static snapshots generated as of that date. Any recency/age calculation (e.g. "how old is this account's snapshot") should use 2026-08-01 as the reference point, not your actual system clock.

## What's here

Provided with the exercise and unmodified — the descriptions below are the scaffold's own:

- `model/model.pkl`, a real, already-trained scikit-learn pipeline. Don't retrain it. You don't need to audit it to research rigor, this exercise isn't scored on that, but it's real data worth actually looking at if it changes your impact framing or monitoring design.
- `data/training_data.csv`, the labeled historical data the model above was actually trained on. Look at it enough to ground your impact-framing numbers and your monitoring design, that's the bar, not a full audit.
- `data/accounts_to_score.csv`, an unlabeled batch you'll run the model against as part of the agent build. Don't modify or regenerate either CSV; everyone works from the same files.
Everything below this line is mine:

- `agent/` — the call-list agent. A LangGraph `StateGraph`: scores the batch, tiers it, flags
  the accounts the model over-scores, cuts to rep capacity, writes a per-account rationale
  through a swappable LLM backend, checks that rationale before a rep can see it, and renders
  the artifacts. `agent/mocks.py` holds both mocked integrations and nothing else does.
- `monitoring/` — two layers above the agent's own input gate: `drift_monitor.py` compares
  runs to each other with CUSUM, `outcome_monitor.py` tests whether the top tier actually
  converted and shows why that check is too slow to be the alarm.
- `analysis/` — the scripts every figure in `PROPOSAL.md` comes from. Re-run them and the
  numbers reproduce.
- `PROPOSAL.md` — impact framing, agent design, monitoring design.
- `RESEARCH-LOG.md` — written as the work happened. The last entry consolidates every number,
  assumption and gap; Appendix A quotes the prompts that changed a decision.
- `CLAUDE.md` — the working context the build ran against: constraints, roadmap, and the
  conventions applied throughout.
- `output/` — a default (mocked) run. `output_groq/` — the same run against a real model via
  Groq, kept as evidence the live path works and as the comparison point for the guardrail.

## Running it

Nothing here needs an API key. The rationale step defaults to a deterministic mock, and both
mocked integrations live in `agent/mocks.py` so they are easy to find and judge. Step 2b is
optional and is the only part that talks to a network.

**1. Reproduce the numbers.** Every figure quoted in `PROPOSAL.md` comes from these:

    python analysis/explore_data.py        # what the data says, incl. tests that failed
    python analysis/score_and_evaluate.py  # is the model worth using, and by how much

**2. Run the agent.** Builds a prioritised call list from `data/accounts_to_score.csv`:

    pip install -r requirements-agent.txt  # adds langgraph on top of the pinned core
    python -m agent.graph

    python -m agent.graph --capacity 40    # how many accounts one rep can work
    python -m agent.graph --print-graph    # print the compiled graph as mermaid

Writes to `output/`:

| File | What it is |
|---|---|
| `call_sheet.md` | What a rep opens. Ordered, cut to capacity, one short brief per account. |
| `crm_tasks.json` | Salesforce Task payloads — the action attached to the list. Mocked at the boundary. |
| `run_report.json` | Input-gate findings, tier counts, score distribution, trace spans, guardrail results. |
| `agent_scored_accounts.csv` | All 300 accounts with score, tier and vendor-data flag. |

`run_report.json` **is** the trace — one span per graph node, plus a child span per LLM call
and per guardrail check. Tracing is local on purpose: the run reproduces from a clean clone
with no account and no key. Setting `LANGSMITH_TRACING`, `LANGSMITH_API_KEY` and
`LANGSMITH_PROJECT` streams the node spans to a hosted backend with no code change; the LLM
and guardrail sub-spans come from the `Tracer` in `agent/mocks.py` and would each need a
`@traceable` decorator to appear there too.

**2b. Swap the LLM backend (optional).** The rationale step is pluggable. `mock` is the
default and needs no key, so everything above runs unchanged without one.

    pip install groq
    setx GROQ_API_KEY "..."                          # or export, per OS

    python -m agent.graph --list-models groq         # what the provider serves today
    python -m agent.graph --llm groq
    python -m agent.graph --llm groq --model openai/gpt-oss-20b
    python -m agent.graph --compare mock,groq        # same accounts, both backends

Providers: `mock` (default), `groq` (tested), `anthropic` and `openai` (written to the
documented API shapes but **untested** — no keys were available). Keys come from the
environment at call time, never from a file. A live provider with no key **fails at startup**
rather than quietly falling back to the mock, because reporting mock output as a real model's
is exactly the kind of silent wrongness this project is about.

`--compare` is the reason this exists. The guardrail does not know which model produced a
rationale — it checks the text against the facts that went into the prompt — so swapping the
backend scores each model on identical inputs against identical criteria:

    backend                                 briefs   passed   rejected   call errors
    mock:deterministic-template (mocked)    25       25       0          0
    groq:openai/gpt-oss-120b                25       21       4          0

**3. Run the guardrail's own tests.** The checks on the LLM's output are validated in both
directions — that they accept good output and reject each specific failure mode:

    python -m agent.guardrails

**4. Run the monitoring checks.**

    python -m monitoring.drift_monitor --simulate   # leading indicators, across runs
    python -m monitoring.outcome_monitor            # the lagging business-outcome check

    python -m monitoring.drift_monitor --history output/   # against real run reports

### How the monitoring is split, and why

There are two layers, watching for different things on different timescales.

**The input gate** (inside the agent) inspects one batch and decides go/no-go. It catches a
cliff: an unrecognised industry, duplicate IDs, a coverage collapse. It is also structurally
blind to slow decline — vendor coverage on 300 accounts has a ~2.8pp standard error, so a
3-sigma gate needs ~8.4pp of movement, and a vendor shedding 0.8pp per run stays invisible to
it for ten weeks.

**`drift_monitor.py`** compares runs to each other using CUSUM, which accumulates small
deviations instead of testing each in isolation. On the bundled simulation, vendor-coverage
drift is caught at run 15 while the single-run check never fires once across 18 runs. Noise is
estimated from a burn-in period rather than assumed, because sampling error is a floor on
real batch-to-batch variation, not an estimate of it — assuming otherwise produces an alarm
that fires weekly and therefore gets muted.

**`outcome_monitor.py`** tests the actual claim: did Tier A convert at ~26.7%? It also shows
why it cannot be the primary alarm. Detecting a fall from 26.7% to 20% needs ~255 Tier A
accounts; at ~28 per batch plus the 90-day conversion window, that is about **five months even
at weekly cadence**. That is the two-quarter gap in the brief's own backstory — an
organisation watching only conversions would reproduce that failure while doing nothing wrong.
The leading indicators exist because this one is too slow, and this one exists because the
leading indicators cannot prove the model became wrong.

### How the agent is put together

A LangGraph `StateGraph` with linear/DAG edges and exactly one branch:

    validate_input ─┬─(proceed)→ score_accounts → assign_tiers → flag_vendor_gap
                    │            → select_call_list → generate_rationales
                    │            → validate_rationales → render_outputs
                    └─(halt)───→ halt_run

Every decision that reaches a rep — ranking, tiering, flagging, the capacity cut, and whether
the batch is fit to score at all — is made by deterministic code and is reproducible across
runs. The LLM sits at a leaf: it writes the justification for a list that is already final,
and its output is checked before a rep can see it.

`validate_input` will **refuse to produce a call list** rather than emit one it cannot stand
behind — on an unrecognised industry (which the encoder would otherwise silently zero),
duplicate account IDs, or a vendor-coverage shift large enough that it cannot be sampling
noise. That is the anti-silent-failure mechanism, and it is the reason the graph has a branch.

## Working process

Commit as you actually go, small, real commits over time, not one commit at the end. We read the commit history as part of how you reason and work, not just the final diff.

**We'd genuinely like you to use AI here, assisted coding tools especially (Claude Code, Codex, Cursor, Antigravity, or similar), on your own accounts.** Dialpad doesn't provide one for this exercise. Disclose your actual sessions/prompts in `RESEARCH-LOG.md`, specific enough that we can see what shaped a decision, not a vague "used AI throughout."

## When you're done

Push this to a public git repo and send us the link. That's the submission. The presentation gets scheduled as a separate follow-up after that, not something to prepare beforehand.
