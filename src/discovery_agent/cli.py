"""Command-line entry point: `discovery-agent <command>`."""

import argparse
import json

from discovery_agent.evaluation.grader import grade_run
from discovery_agent.graph.workflow import run_discovery
from discovery_agent.healthcheck import run_checks
from discovery_agent.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(prog="discovery-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Fetch ArXiv/PubMed papers into Milvus")
    ingest.add_argument("--query", required=True)
    ingest.add_argument("--max-results", type=int, default=None)

    run = sub.add_parser("run", help="Run the full discovery loop on a research topic")
    run.add_argument("--topic", required=True)
    run.add_argument("--thread-id", default=None, help="Resume a previous run")

    grade = sub.add_parser("grade", help="Evaluate a completed run against the literature holdout set")
    grade.add_argument("--run-dir", required=True, help="Path to the completed run directory")
    grade.add_argument("--holdout", default="data/holdout.jsonl", help="Holdout JSONL path")

    sub.add_parser("check", help="Verify config, Milvus, Docker sandbox and LLM connectivity")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion(args.query, max_results=args.max_results)
    elif args.command == "run":
        run_discovery(args.topic, thread_id=args.thread_id)
    elif args.command == "grade":
        print(json.dumps(grade_run(args.run_dir, args.holdout), indent=2))
    elif args.command == "check":
        run_checks()


if __name__ == "__main__":
    main()
