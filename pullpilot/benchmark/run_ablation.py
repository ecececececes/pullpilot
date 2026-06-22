"""Context-grounding ablation for the LLM engine: with-context vs no-context,
same model, same PRs. This is the project's central result (needs an API key).

    python -m pullpilot.benchmark.run_ablation --provider anthropic
    python -m pullpilot.benchmark.run_ablation --provider mock   # wiring test
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

from ..engines import LLMEngine
from ..providers import get_provider
from ..reviewer import PullRequest, Reviewer
from .build_dataset import load_dataset
from .evaluate import Metrics, evaluate

_DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "examples", "dataset.json"
)


def run(dataset_path: str, provider_name: str) -> Dict[str, Metrics]:
    prs = load_dataset(dataset_path)
    results: Dict[str, Metrics] = {}
    for use_ctx in (False, True):
        engine = LLMEngine(get_provider(provider_name))
        reviewer = Reviewer(engine, use_context=use_ctx)
        reviews = {
            p.id: reviewer.review(
                PullRequest(diff=p.diff, post_files={p.file: p.post_source},
                            title=p.title, description=p.description)
            )
            for p in prs
        }
        results["with_context" if use_ctx else "no_context"] = evaluate(prs, reviews)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=_DEFAULT_DATASET)
    ap.add_argument("--provider", default="mock")
    args = ap.parse_args()

    res = run(args.dataset, args.provider)
    nc, wc = res["no_context"], res["with_context"]

    print(f"\nContext-grounding ablation  (provider = {args.provider})")
    print("-" * 64)
    print(f"{'metric':<30}{'no_context':>16}{'with_context':>16}")
    print("-" * 64)
    for label, attr in [
        ("detection recall", "recall"),
        ("precision", "precision"),
        ("exact localization", "exact_localization"),
        ("false alarms / clean PR", "false_alarms_per_clean_pr"),
    ]:
        print(f"{label:<30}{getattr(nc, attr):>16.3f}{getattr(wc, attr):>16.3f}")
    print("-" * 64)
    print(f"buggy PRs = {nc.n_buggy}   clean PRs = {nc.n_clean}")
    if args.provider == "mock":
        print("\n[note] mock ignores context, so columns match. Use a real provider.")


if __name__ == "__main__":
    main()
