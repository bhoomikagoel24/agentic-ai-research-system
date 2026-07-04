"""
main_langgraph.py

v2 entry point. Deliberately a separate file from main.py — v1 stays
exactly as it is, runnable and demoable, while this exercises the
LangGraph pipeline.

Usage:
    python main_langgraph.py

Notable differences from main.py's run_pipeline():
  - No on_step callback plumbing. graph.stream(..., stream_mode="values")
    natively yields the running state after every node, which is what
    on_step was emulating by hand.
  - thread_id ties this run to a LangGraph checkpoint, so if the
    process dies mid-run you can re-invoke with the same config and
    LangGraph resumes from the last completed node instead of
    restarting from "planner".
"""

import sys
import json
import hashlib

from research_agent_system.config.config import Config
from research_agent_system.workflows.langgraph_workflow import build_graph

sys.stdout.reconfigure(encoding="utf-8")


def main():
    topic = input("Enter research topic: ").strip()
    if not topic:
        topic = "GenAI for stock prediction"

    cfg = Config(dev_mode=False)
    graph = build_graph(cfg)

    thread_id = hashlib.md5(topic.encode("utf-8")).hexdigest()
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nRunning pipeline for: {topic}\n")

    final_state = {}
    try:
        for step_state in graph.stream({"topic": topic}, config=config, stream_mode="values"):
            done_keys = [k for k in step_state if step_state.get(k) not in (None, [], {}, "")]
            print(f"... pipeline state now includes: {done_keys}")
            final_state = step_state
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        return

    print("\n📊 METRICS:\n")
    papers = final_state.get("papers", [])
    summaries = final_state.get("summaries", [])
    print(json.dumps({
        "total_papers": len(papers),
        "total_summaries": len(summaries),
        "critique_retries": final_state.get("critique_retries", 0),
    }, indent=2))

    print("\n📄 FINAL REPORT:\n")
    print(final_state.get("final_report", "No report generated."))
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
