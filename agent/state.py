"""
The object that flows through the graph.

WHY A TYPED STATE RATHER THAN PASSING A DATAFRAME AROUND: each node declares
what it adds, so the order of operations is readable from the type alone and a
node cannot quietly depend on something an earlier node forgot to set. It is
also what makes the graph extensible later without rewriting the call chain.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class AgentState(TypedDict, total=False):
    # --- inputs, set before the graph runs ---
    input_csv: str
    capacity: int          # how many accounts a rep can realistically work
    output_dir: str

    # --- set by validate_input ---
    gate: dict[str, Any]   # findings, severity, and the halt decision
    halted: bool

    # --- set by score_accounts / assign_tiers / flag_vendor_gap ---
    accounts: pd.DataFrame  # all 300, scored, tiered, flagged

    # --- set by select_call_list ---
    call_list: pd.DataFrame  # the cut, ordered slice a rep actually works

    # --- set by generate_rationales / validate_rationales ---
    briefs: list[dict[str, Any]]

    # --- set by render_outputs / halt_run ---
    artifacts: dict[str, str]
    report: dict[str, Any]
