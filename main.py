# =======================================================================================================================================================================================================================================================================
#                                     END-To-End  AGENTIC RESEARCH SYNTHESIS SYSTEM
# =======================================================================================================================================================================================================================================================================
import sys
import json

from research_agent_system.config.config import (
    Config
)

from research_agent_system.workflows.research_pipeline import (
    run_pipeline,
    evaluate_pipeline
)


# UTF-8 
sys.stdout.reconfigure(encoding="utf-8")

# MAIN

def main():

    topic = input("Enter research topic: ").strip()

    if not topic:
        topic = ("GenAI for stock prediction")

    # CONFIG
    cfg = Config(dev_mode=False)

    # RUN PIPELINE
    try:

        state = run_pipeline(topic,cfg)

    except Exception as e:

        print(f"\nPipeline failed: {e}")

        return

    # METRICS
    metrics = evaluate_pipeline(state)

    print("\n📊 METRICS:\n")
    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False
        )
    )

    # FINAL REPORT
    print("\n📄 FINAL REPORT:\n")
    print(state.final_report)
    print("\nPipeline completed successfully.")

# ENTRY
if __name__ == "__main__":
    main()