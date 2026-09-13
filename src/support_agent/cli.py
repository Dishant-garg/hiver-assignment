"""Tiny CLI: `python -m support_agent.cli demo "message"` prints the decision as JSON."""
import argparse
import json
import sys

from support_agent import config
from support_agent.agent import SupportAgent
from support_agent.llm import CacheMissError, LLMClient
from support_agent.retrieve import Retriever


def main() -> None:
    p = argparse.ArgumentParser(prog="support_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="Run the agent on one message")
    d.add_argument("message")
    args = p.parse_args()
    try:
        agent = SupportAgent(LLMClient(), Retriever.from_csv(config.brand_dir() / "corpus.csv"))
        print(json.dumps(agent.handle(args.message).to_dict(), indent=2, ensure_ascii=False))
    except (CacheMissError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
