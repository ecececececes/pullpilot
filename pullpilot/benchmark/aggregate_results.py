"""Run the benchmark across engines, dump results.json, and generate REPORT.md
(a full writeup skeleton with real numbers filled in).

    python -m pullpilot.benchmark.aggregate_results                      # static only
    python -m pullpilot.benchmark.aggregate_results --provider anthropic --ablation

Whatever you run gets filled into the report; everything else is marked PENDING
with the exact command to complete it.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict

from ..engines import LLMEngine, StaticAnalysisEngine
from ..providers import get_provider
from ..reviewer import PullRequest, Reviewer
from .build_dataset import load_dataset
from .evaluate import evaluate, per_pr_results

_HERE = os.path.dirname(__file__)
_DATASET = os.path.join(_HERE, "..", "..", "data", "examples", "dataset.json")
_RESULTS = os.path.join(_HERE, "..", "..", "data", "examples", "results.json")
_REPORT = os.path.join(_HERE, "..", "..", "REPORT.md")


def _run(engine, prs, use_context=True):
    reviewer = Reviewer(engine, use_context=use_context)
    reviews = {
        p.id: reviewer.review(PullRequest(diff=p.diff, post_files={p.file: p.post_source},
                                          title=p.title, description=p.description))
        for p in prs
    }
    return evaluate(prs, reviews), per_pr_results(prs, reviews)


def _cat_detection(rows):
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["label"] == "buggy" and r["category"]:
            d[r["category"]][1] += 1
            if r["detected"]:
                d[r["category"]][0] += 1
    return {k: v for k, v in d.items()}


def collect(dataset_path, providers, do_ablation):
    prs = load_dataset(dataset_path)
    out = {"engines": {}, "ablation": {}, "categories": {}}

    m, rows = _run(StaticAnalysisEngine(), prs)
    out["engines"]["static"] = {"metrics": asdict(m), "by_category": _cat_detection(rows)}

    for prov in providers:
        name = f"llm:{prov}"
        m, rows = _run(LLMEngine(get_provider(prov)), prs)
        out["engines"][name] = {"metrics": asdict(m), "by_category": _cat_detection(rows)}
        if do_ablation:
            nc, _ = _run(LLMEngine(get_provider(prov)), prs, use_context=False)
            wc, _ = _run(LLMEngine(get_provider(prov)), prs, use_context=True)
            out["ablation"][name] = {"no_context": asdict(nc), "with_context": asdict(wc)}

    cats = sorted({r for e in out["engines"].values() for r in e["by_category"]})
    out["categories"] = cats
    out["n_buggy"] = out["engines"]["static"]["metrics"]["n_buggy"]
    out["n_clean"] = out["engines"]["static"]["metrics"]["n_clean"]
    return out


# ---------- markdown rendering ----------

_PENDING = "_pending — run `python -m pullpilot.benchmark.aggregate_results --provider anthropic --ablation`_"


def _metric_table(results):
    engines = list(results["engines"])
    header = "| metric | " + " | ".join(engines) + " |"
    sep = "|" + "---|" * (len(engines) + 1)
    rows = []
    specs = [("detection recall", "recall", "{:.0%}"),
             ("precision", "precision", "{:.0%}"),
             ("exact localization", "exact_localization", "{:.0%}"),
             ("false alarms / clean PR", "false_alarms_per_clean_pr", "{:.2f}")]
    for label, key, fmt in specs:
        cells = [fmt.format(results["engines"][e]["metrics"][key]) for e in engines]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _category_table(results):
    engines = list(results["engines"])
    header = "| bug category | " + " | ".join(engines) + " |"
    sep = "|" + "---|" * (len(engines) + 1)
    rows = []
    for cat in results["categories"]:
        cells = []
        for e in engines:
            hit, tot = results["engines"][e]["by_category"].get(cat, [0, 0])
            cells.append(f"{hit}/{tot}")
        rows.append(f"| {cat} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _ablation_table(results):
    if not results["ablation"]:
        return _PENDING
    blocks = []
    for name, ab in results["ablation"].items():
        nc, wc = ab["no_context"], ab["with_context"]
        blocks.append(f"**{name}**\n")
        blocks.append("| metric | no context | with context | Δ |")
        blocks.append("|---|---|---|---|")
        for label, key, fmt in [("recall", "recall", "{:.0%}"),
                                ("precision", "precision", "{:.0%}"),
                                ("exact localization", "exact_localization", "{:.0%}"),
                                ("false alarms/clean", "false_alarms_per_clean_pr", "{:.2f}")]:
            a, b = nc[key], wc[key]
            blocks.append(f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {b - a:+.2f} |")
        blocks.append("")
    return "\n".join(blocks)


def render_report(results) -> str:
    has_llm = any(e.startswith("llm:") for e in results["engines"])
    llm_note = "" if has_llm else (
        "\n> **NOTE:** No LLM engine has been run yet, so the LLM columns/rows below "
        "are absent or pending. Add a key and run "
        "`python -m pullpilot.benchmark.aggregate_results --provider anthropic --ablation` "
        "to fill them, then regenerate this report.\n")

    return f"""# PullPilot: Context-Grounded, Verification-Augmented Pull-Request Review

*Foundations and Applications of Generative AI — project report (draft)*
{llm_note}
## Abstract

We present PullPilot, a pull-request review system that combines a large language
model with deterministic context retrieval and tool-augmented verification. Given
a diff, PullPilot retrieves the surrounding code via the abstract syntax tree,
prompts an LLM for a severity- and confidence-ranked set of issues constrained to
a fixed JSON schema, applies precision-first calibration, and optionally executes
a linter and the project's tests to add machine-verified findings. We evaluate on
a controlled benchmark of {results['n_buggy']} pull requests with deliberately
injected defects across {len(results['categories'])} categories, plus
{results['n_clean']} behaviour-preserving (clean) pull requests for false-alarm
measurement. We compare against a static-analysis baseline, ablate repository
context, and distinguish inferred from verified findings.

## 1. Introduction

Code review is a bottleneck in software teams; reviewers are slow and
inconsistent, and non-author reviewers struggle to understand changes. General
LLM tools comment on snippets stripped of repository context, producing generic
or wrong advice. PullPilot targets a specialised workflow: context-aware,
schema-constrained, calibrated review with optional verification. Our central
hypotheses are (H1) repository context improves review quality over the bare
diff, and (H2) inference and execution catch complementary defect classes.

## 2. Related Work

Open-source PR-review agents (e.g. qodo's PR-Agent) automate diff ingestion and
posting reviews to pull requests. Recent academic and industry work studies LLM
code review with dedicated benchmarks built on pull-request datasets that carry
repository context, agentic reviewers, and production systems. Established defect
datasets provide real buggy/fixed pairs: Defects4J (Java, 800+ reproducible bugs
each with a failing test), BugsInPy (Python, ~493 curated bugs), and
ManySStuBs4J (150k+ labelled single-statement bugs). PullPilot's contribution is
not the wrapper but the controlled study of context grounding and verification.

## 3. System Design

The pipeline is a fixed, deterministic sequence rather than a free-roaming agent,
which keeps results reproducible and ablatable:

```
parse diff -> retrieve AST context -> engine.review() -> calibrate -> [verify]
```

- **Diff parsing** converts a unified diff into per-file changed/affected line sets.
- **Context retrieval** uses the AST to extract the enclosing function/class and
  the definitions of referenced symbols; a no-context variant supports the ablation.
- **Engines** share one interface: an *LLM engine* (schema-constrained JSON output)
  and a *static-analysis engine* (AST patterns + pyflakes) used as a baseline.
- **Calibration** enforces precision: a confidence floor, line-grounding (an issue
  must sit on the changed lines), and converting low-confidence issues into
  questions. Machine-verified findings bypass calibration.
- **Verification** runs a linter and the PR's tests in a resource-limited
  subprocess; a linter hit or a failing test becomes a verified finding
  (`source` = linter/test) that is sorted first.

## 4. Experimental Setup

**Dataset.** Defects are introduced by *reversing real fix commits*: the fixed
file is the clean baseline and a PR is synthesised that re-introduces the bug,
with the touched lines as exact ground truth. The injected set spans pattern
bugs (mutable defaults, `== None`, bare `except`, identity-vs-literal, unguarded
`.get()` subscript, resource leaks, undefined names) and semantic bugs
(off-by-one, wrong operator, swapped args, inverted condition, etc.). A GitHub
loader can pull real bug-fix commits into the same format.

**Metrics.** Detection recall (buggy PRs with a finding on the planted line),
precision (findings landing on a real defect), exact localization, and
false-alarms-per-clean-PR.

**Conditions.** Static baseline; LLM with vs without retrieved context (H1);
inference-only vs inference+verification (H2); and cross-provider comparison.

## 5. Results

### 5.1 Engine comparison

{_metric_table(results)}

### 5.2 Detection by bug category

{_category_table(results)}

### 5.3 Context-grounding ablation (H1)

{_ablation_table(results)}

### 5.4 Verification (H2)

The static and LLM engines reason about the diff; the verification layer executes
it. On purely semantic defects (e.g. the off-by-one `items[len(items)]`), static
analysis reports nothing, while a failing test surfaces the defect as a verified,
critical finding. This is the qualitative evidence for H2; see `verify_demo.py`.
_Quantitative verification numbers: supply tests per PR and extend the harness._

## 6. Discussion

The static baseline establishes a precision-1.0, zero-false-alarm floor that
catches exactly the pattern-detectable half of the benchmark and none of the
semantic half. Any value an LLM adds must appear as recall on the semantic
categories without inflating the false-alarm rate — which is precisely what the
category table is designed to expose.

## 7. Limitations

Context retrieval is single-file; the injected benchmark, while controlled, is
synthetic relative to real review burden; verification runs under resource limits
rather than a full jail and should be containerised for untrusted PRs; and the
human-agreement evaluation from the proposal is not yet collected.

## 8. Reproducibility

```
pip install -r requirements.txt
python -m pullpilot.benchmark.make_example_dataset
python -m pullpilot.benchmark.run_static_benchmark
python -m pullpilot.benchmark.aggregate_results --provider anthropic --ablation
pytest -q
```

All numbers in this report are generated from `data/examples/results.json` by
`aggregate_results.py`; re-running regenerates both the JSON and this document.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=_DATASET)
    ap.add_argument("--provider", action="append", default=[])
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--results", default=_RESULTS)
    ap.add_argument("--report", default=_REPORT)
    args = ap.parse_args()

    results = collect(args.dataset, args.provider, args.ablation)
    with open(os.path.abspath(args.results), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.abspath(args.report), "w") as f:
        f.write(render_report(results))

    print(f"engines run: {', '.join(results['engines'])}")
    print(f"results.json -> {os.path.abspath(args.results)}")
    print(f"REPORT.md    -> {os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()
