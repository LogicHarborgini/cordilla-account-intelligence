"""
Graph wiring and CLI entry point.

    python -m agent.graph                          # score the provided batch
    python -m agent.graph --capacity 40
    python -m agent.graph --input path/to.csv --output-dir output/

WHY LANGGRAPH FOR A PIPELINE THIS LINEAR - the honest version, since it is a
fair thing to challenge:

  * It is NOT because the control flow demands it. There is one branch (the
    input gate) and no cycles. Plain function calls would run this correctly.
  * It IS because the node boundaries become the observability boundaries.
    Every node is a span; the run report is assembled from them; the
    monitoring work reads that report. With LANGSMITH_TRACING set, the same
    graph emits the same spans to a hosted backend with no code change.
  * And because the extension this product would plausibly get next -
    per-account research where each step depends on the last - is new nodes
    and conditional edges here, rather than a rewrite.

What was deliberately NOT adopted: cycles, LLM-directed routing, and agent
executors. Every decision that reaches a sales rep is made by deterministic
code and is reproducible across runs. Using LangGraph's agentic machinery here
would trade that away for the appearance of using the framework properly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.state import AgentState

DEFAULT_INPUT = "data/accounts_to_score.csv"
DEFAULT_OUTPUT_DIR = "output"

# How many accounts one rep works in a sitting. ASSUMPTION, NOT A FINDING:
# rep headcount and call volume are absent from the provided data. 25 is a
# plausible day's outbound and happens to sit just under the 28 Tier A
# accounts in the supplied batch, which keeps the cut meaningful.
DEFAULT_CAPACITY = 25


def build_graph():
    """Wire the nodes into a DAG. Edges are the design; read them top to bottom."""
    builder = StateGraph(AgentState)

    builder.add_node("validate_input", nodes.validate_input)
    builder.add_node("score_accounts", nodes.score_accounts)
    builder.add_node("assign_tiers", nodes.assign_tiers)
    builder.add_node("flag_vendor_gap", nodes.flag_vendor_gap)
    builder.add_node("select_call_list", nodes.select_call_list)
    builder.add_node("generate_rationales", nodes.generate_rationales)
    builder.add_node("validate_rationales", nodes.validate_rationales)
    builder.add_node("render_outputs", nodes.render_outputs)
    builder.add_node("halt_run", nodes.halt_run)

    builder.add_edge(START, "validate_input")

    # The only branch in the graph, and it is decided by a boolean the gate
    # computed from the data - never by a model.
    builder.add_conditional_edges(
        "validate_input",
        nodes.route_after_gate,
        {"proceed": "score_accounts", "halt": "halt_run"},
    )

    builder.add_edge("score_accounts", "assign_tiers")
    builder.add_edge("assign_tiers", "flag_vendor_gap")
    builder.add_edge("flag_vendor_gap", "select_call_list")
    builder.add_edge("select_call_list", "generate_rationales")
    builder.add_edge("generate_rationales", "validate_rationales")
    builder.add_edge("validate_rationales", "render_outputs")
    builder.add_edge("render_outputs", END)
    builder.add_edge("halt_run", END)

    return builder.compile()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a prioritised SDR call list.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY,
                        help="accounts one rep can work (assumption, not from the data)")
    parser.add_argument("--print-graph", action="store_true",
                        help="print the compiled graph as mermaid and exit")
    args = parser.parse_args()

    graph = build_graph()

    if args.print_graph:
        print(graph.get_graph().draw_mermaid())
        return 0

    final = graph.invoke({
        "input_csv": args.input,
        "output_dir": args.output_dir,
        "capacity": args.capacity,
    })

    gate = final["gate"]
    print(f"input        : {args.input} ({gate['rows_in']} rows)")
    print(f"gate         : {gate['counts']['halt']} halt, "
          f"{gate['counts']['warn']} warn, {gate['counts']['info']} info")
    for finding in gate["findings"]:
        if finding["severity"] != "info":
            print(f"   [{finding['severity'].upper()}] {finding['code']}: {finding['detail']}")

    if final.get("halted"):
        print("\nHALTED - input failed validation. No call list written.")
        print(f"see {Path(args.output_dir) / 'run_report.json'}")
        return 1

    report = final["report"]
    print(f"tiers        : {report['scoring']['tier_counts']}")
    print(f"call list    : {len(final['call_list'])} accounts (capacity {args.capacity})")
    print(f"rationales   : {report['llm']['by_source']}")
    print(f"guardrail    : {report['llm']['guardrail_rejections']} rejected"
          f"{' -> ' + json.dumps(report['llm']['guardrail_failure_codes']) if report['llm']['guardrail_failure_codes'] else ''}")
    print(f"trace        : {report['trace']['span_count']} spans "
          f"({report['trace']['run_id']})")
    print("\nwrote:")
    for name, path in final["artifacts"].items():
        print(f"   {name:<18} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
