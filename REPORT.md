# PullPilot: Context-Grounded, Verification-Augmented Pull-Request Review

*Foundations and Applications of Generative AI — project report (draft)*

> **NOTE:** No LLM engine has been run yet, so the LLM columns/rows below are absent or pending. Add a key and run `python -m pullpilot.benchmark.aggregate_results --provider anthropic --ablation` to fill them, then regenerate this report.

## Abstract

We present PullPilot, a pull-request review system that combines a large language
model with deterministic context retrieval and tool-augmented verification. Given
a diff, PullPilot retrieves the surrounding code via the abstract syntax tree,
prompts an LLM for a severity- and confidence-ranked set of issues constrained to
a fixed JSON schema, applies precision-first calibration, and optionally executes
a linter and the project's tests to add machine-verified findings. We evaluate on
a controlled benchmark of 20 pull requests with deliberately
injected defects across 20 categories, plus
8 behaviour-preserving (clean) pull requests for false-alarm
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

| metric | static |
|---|---|
| detection recall | 50% |
| precision | 100% |
| exact localization | 92% |
| false alarms / clean PR | 0.00 |

### 5.2 Detection by bug category

| bug category | static |
|---|---|
| and_or_mixup | 0/1 |
| bare_except | 1/1 |
| bare_except_2 | 1/1 |
| eq_none | 1/1 |
| inverted_condition | 0/1 |
| is_literal | 1/1 |
| missing_return | 0/1 |
| mutable_default | 1/1 |
| mutable_default_dict | 1/1 |
| neq_none | 1/1 |
| none_subscript | 1/1 |
| off_by_one | 0/1 |
| off_by_one_slice | 0/1 |
| resource_leak | 1/1 |
| sign_error | 0/1 |
| swapped_args | 0/1 |
| undefined_name | 1/1 |
| wrong_boundary | 0/1 |
| wrong_index | 0/1 |
| wrong_operator | 0/1 |

### 5.3 Context-grounding ablation (H1)

_pending — run `python -m pullpilot.benchmark.aggregate_results --provider anthropic --ablation`_

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
