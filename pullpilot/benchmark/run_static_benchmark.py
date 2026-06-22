"""Run the REAL static-analysis reviewer over the benchmark. No API key needed.

    python -m pullpilot.benchmark.run_static_benchmark

Produces real recall/precision/localization/false-alarm numbers, a per-category
breakdown, and an HTML report at data/examples/report.html.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

from ..engines import StaticAnalysisEngine
from ..report import render_report
from ..reviewer import PullRequest, Reviewer
from .build_dataset import load_dataset
from .evaluate import _issue_hits, evaluate, per_pr_results

_DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples", "dataset.json")
_REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples", "report.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=_DATASET)
    ap.add_argument("--report", default=_REPORT)
    args = ap.parse_args()

    prs = load_dataset(args.dataset)
    reviewer = Reviewer(StaticAnalysisEngine(), use_context=True)
    reviews = {
        p.id: reviewer.review(
            PullRequest(diff=p.diff, post_files={p.file: p.post_source},
                        title=p.title, description=p.description)
        )
        for p in prs
    }

    metrics = evaluate(prs, reviews)
    rows = per_pr_results(prs, reviews)

    print("\nStatic-analysis reviewer — defect-detection benchmark")
    print("-" * 56)
    print(f"detection recall          {metrics.recall:6.1%}  ({metrics.detected}/{metrics.n_buggy})")
    print(f"precision                 {metrics.precision:6.1%}")
    print(f"exact localization        {metrics.exact_localization:6.1%}")
    print(f"false alarms / clean PR   {metrics.false_alarms_per_clean_pr:6.2f}  ({metrics.n_clean} clean PRs)")
    print("-" * 56)

    # per-category detection
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["label"] == "buggy":
            by_cat[r["category"]][1] += 1
            if r["detected"]:
                by_cat[r["category"]][0] += 1
    print("detection by category:")
    for cat in sorted(by_cat):
        hit, tot = by_cat[cat]
        mark = "OK " if hit else "-- "
        print(f"  {mark}{cat:<22}{hit}/{tot}")
    print("-" * 56)

    path = render_report("PullPilot — Static Analysis Benchmark", metrics, rows, os.path.abspath(args.report))
    print(f"HTML report: {path}")


if __name__ == "__main__":
    main()
