"""Run several review engines over the same benchmark and compare them.

Always runs the offline static analyser. Add LLM providers to compare:

    python -m pullpilot.benchmark.run_comparison                       # static only
    python -m pullpilot.benchmark.run_comparison --provider anthropic  # + LLM
    python -m pullpilot.benchmark.run_comparison --provider anthropic --provider openai

Writes data/examples/comparison.html and prints a metric table.
"""
from __future__ import annotations

import argparse
import os

from ..engines import LLMEngine, StaticAnalysisEngine
from ..providers import get_provider
from ..report import render_comparison
from ..reviewer import PullRequest, Reviewer
from .build_dataset import load_dataset
from .evaluate import evaluate, per_pr_results

_DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples", "dataset.json")
_OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples", "comparison.html")


def _run_engine(engine, prs):
    reviewer = Reviewer(engine, use_context=True)
    reviews = {
        p.id: reviewer.review(
            PullRequest(diff=p.diff, post_files={p.file: p.post_source},
                        title=p.title, description=p.description)
        )
        for p in prs
    }
    return evaluate(prs, reviews), per_pr_results(prs, reviews)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=_DATASET)
    ap.add_argument("--out", default=_OUT)
    ap.add_argument("--provider", action="append", default=[],
                    help="LLM provider(s) to include; repeatable")
    args = ap.parse_args()

    prs = load_dataset(args.dataset)

    engines = [("static", StaticAnalysisEngine())]
    for prov in args.provider:
        engines.append((f"llm:{prov}", LLMEngine(get_provider(prov))))

    results = []
    for name, engine in engines:
        metrics, rows = _run_engine(engine, prs)
        results.append((name, metrics, rows))

    print("\nEngine comparison")
    print("-" * (30 + 14 * len(results)))
    header = f"{'metric':<26}" + "".join(f"{n:>14}" for n, _, _ in results)
    print(header)
    print("-" * (30 + 14 * len(results)))
    for label, attr in [
        ("detection recall", "recall"),
        ("precision", "precision"),
        ("exact localization", "exact_localization"),
        ("false alarms/clean", "false_alarms_per_clean_pr"),
    ]:
        line = f"{label:<26}" + "".join(
            f"{getattr(m, attr):>14.3f}" for _, m, _ in results
        )
        print(line)
    print("-" * (30 + 14 * len(results)))

    path = render_comparison("PullPilot — Engine Comparison", results, os.path.abspath(args.out))
    print(f"HTML report: {path}")


if __name__ == "__main__":
    main()
