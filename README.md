# PullPilot — context-grounded PR review + working benchmark

A code-review pipeline with **two interchangeable engines** behind one harness:

- **StaticAnalysisEngine** — a real AST + pyflakes detector. **Works offline, no
  API key**, and produces real benchmark numbers. Doubles as the linter baseline.
- **LLMEngine** — prompt → structured JSON review (Anthropic / OpenAI / mock).

The pipeline is a fixed, deterministic sequence (not a free-roaming agent) so
results are reproducible and ablatable:

```
parse diff -> retrieve AST context -> engine.review(...) -> calibrate (precision-first)
```

## Quickstart (runs offline, real results, no key)

```bash
pip install -r requirements.txt
python -m pullpilot.benchmark.make_example_dataset   # 28 PRs (20 buggy / 8 clean)
python demo.py                                       # one real static review
python -m pullpilot.benchmark.run_static_benchmark   # full benchmark + HTML report
python verify_demo.py                                # tool-augmented verification
pytest -q                                            # 11 tests
```

### Actual benchmark output (static engine, 28 PRs)

```
detection recall           50.0%  (10/20)
precision                 100.0%
exact localization         91.7%
false alarms / clean PR     0.00  (8 clean PRs)
```

The analyser catches **all 10 pattern-based bugs** (mutable defaults, `== None`,
bare `except`, identity-vs-literal, unguarded `.get()` subscript, resource leaks,
undefined names) at 100% precision and **0 false alarms**, and **misses all 10
purely semantic bugs** (off-by-one, wrong operator, swapped args, inverted
condition, …). That gap is the experiment: an LLM reviewer should close it.

## Web UI (for the demo)

A local web app: paste a diff, pick an engine, see the review — plus a benchmark
dashboard.

```bash
pip install flask          # already in requirements.txt
python -m pullpilot.web
# open http://localhost:5000
```

The Review page has a dropdown of sample pull requests so it works instantly with
the offline `static` engine (no key). Pick a free engine (`gemini`, `groq`, ...)
to use the LLM — it reads the key from the environment variable in the terminal
you launched it from. The Benchmark page reads `data/examples/results.json`
(created by `aggregate_results`).

## Free providers (no credit card)

Anthropic's API is paid, but the harness speaks any OpenAI-compatible endpoint,
so you can run the whole project on a **free** key. Get your own key (don't use
shared-key sites) and pass the preset as `--provider`:

| preset | where to get a free key | env var |
|---|---|---|
| `gemini` | aistudio.google.com (Google AI Studio) | `GEMINI_API_KEY` |
| `groq` | console.groq.com | `GROQ_API_KEY` |
| `github` | github.com (GitHub Models, any account) | `GITHUB_TOKEN` |
| `ollama` | runs locally, no key, no internet | — |

```bash
export GEMINI_API_KEY=...
python -m pullpilot.benchmark.aggregate_results --provider gemini --ablation
# or: --provider groq / --provider github / --provider ollama
```

Defaults can be overridden if a model name goes stale, e.g.
`GROQ_MODEL=llama-3.1-8b-instant` or `GEMINI_MODEL=gemini-2.5-flash`.
Note: free tiers may use your prompts for training — fine for this open-source
benchmark, but keep private data off them.

## LLM engine + the context ablation (needs a key)

```bash
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
python demo.py llm anthropic
python -m pullpilot.benchmark.run_ablation --provider anthropic   # with vs without context
python -m pullpilot.benchmark.run_ablation --provider openai      # cross-provider
```

The ablation runs the LLM engine **with** retrieved context and **without** it
over the same PRs — the central hypothesis of the project.

### Static vs LLM, side by side

```bash
python -m pullpilot.benchmark.run_comparison                       # static only
python -m pullpilot.benchmark.run_comparison --provider anthropic  # static + LLM
python -m pullpilot.benchmark.run_comparison --provider anthropic --provider openai
```

### One-shot report

```bash
python -m pullpilot.benchmark.aggregate_results --provider anthropic --ablation
```

Runs every engine, writes `data/examples/results.json`, and generates `REPORT.md`
— a full writeup with the live numbers filled in and any unrun parts marked
PENDING with the command to complete them.

## GitHub Action (auto-review live PRs)

PullPilot can run on every pull request and post its review as a comment.
`.github/workflows/pullpilot.yml` runs the reviewer in CI; add repo secrets
`ANTHROPIC_API_KEY` (and the built-in `GITHUB_TOKEN` is provided automatically).

Try it locally first (prints the comment instead of posting):

```bash
export GITHUB_TOKEN=...        # pull-requests:write to post
python -m pullpilot.github_review --owner OWNER --repo REPO --pr 42 \
    --provider anthropic --dry-run
```

Runs every engine over the same PRs and writes `comparison.html` with a metric
table and a per-bug-category detection matrix — the figure for your writeup.

## Tool-augmented verification (machine-verified findings)

Beyond inference, PullPilot can *check*: run a linter and the PR's own tests and
fold the results in as **verified** findings (`source` = `linter` / `test`),
which sort first and bypass calibration because they are facts, not guesses.

```bash
python verify_demo.py
```

In the demo, static analysis reports nothing on an off-by-one
(`items[len(items)]`) — it's purely semantic — but a failing test catches it as
a CRITICAL, verified finding with the real `IndexError`. Supply tests via
`PullRequest(tests=...)` and enable with `Reviewer(engine, verify=True)`.

Execution runs in a temp dir under a wall-clock timeout and POSIX resource
limits (CPU time, output size) with a stripped environment. This is
defence-in-depth, **not** a true jail: for running untrusted internet PRs, wrap
it in a container / nsjail (see `run_tests(..., allow_untrusted=...)`).

## Layout

| File | Role |
|---|---|
| `pullpilot/schema.py` | Fixed Issue/Review JSON contract |
| `pullpilot/diff_parser.py` | Unified diff → structured changed/affected lines |
| `pullpilot/context_retriever.py` | AST context + no-context baseline (for the ablation) |
| `pullpilot/static_analysis.py` | **Real detector** (AST patterns + pyflakes) |
| `pullpilot/engines.py` | `StaticAnalysisEngine`, `LLMEngine` behind one interface |
| `pullpilot/providers.py` | Mock / Anthropic / OpenAI |
| `pullpilot/prompts.py` | System + user prompts (**iterate here for the LLM**) |
| `pullpilot/calibration.py` | Precision-first filter; verified findings bypass it |
| `pullpilot/verification.py` | **Real linter + test execution** (sandboxed), machine-verified findings |
| `pullpilot/reviewer.py` | Pipeline glue |
| `pullpilot/report.py` | HTML report renderer |
| `pullpilot/github_review.py` | **GitHub Action**: fetch a live PR, review, post a comment |
| `pullpilot/web.py` | **Web UI**: live review page + benchmark dashboard (Flask) |
| `pullpilot/benchmark/build_dataset.py` | Reverse fix commits → labelled buggy PRs |
| `pullpilot/benchmark/make_example_dataset.py` | The 28-PR injected-defect set |
| `pullpilot/benchmark/evaluate.py` | recall / precision / localization / false alarms |
| `pullpilot/benchmark/run_static_benchmark.py` | Offline benchmark + HTML report |
| `pullpilot/benchmark/run_ablation.py` | LLM with-context vs no-context |
| `pullpilot/benchmark/run_comparison.py` | Static vs LLM(s), side-by-side report |
| `pullpilot/benchmark/aggregate_results.py` | Runs engines, writes results.json + REPORT.md |
| `pullpilot/benchmark/github_loader.py` | Pull REAL bug-fix commits from GitHub |
| `verify_demo.py` | Tool-augmented verification demo |
| `.github/workflows/pullpilot.yml` | CI workflow that auto-reviews PRs |
| `REPORT.md` | Generated report draft (real numbers + PENDING markers) |
| `tests/test_pullpilot.py` | 23 unit + e2e tests |

## Using REAL bugs (GitHub / BugsInPy / Defects4J)

Two routes, both reverse a fix into a bug-introducing PR with exact labels:

**From a GitHub fix commit** (no manual file wrangling):

```bash
export GITHUB_TOKEN=...   # optional but lifts the 60/hr unauthenticated limit
python -m pullpilot.benchmark.github_loader --owner OWNER --repo REPO \
    --sha FIX_COMMIT_SHA --path path/to/file.py --out data/examples/real.json
python -m pullpilot.benchmark.run_static_benchmark --dataset data/examples/real.json
```

The commit's file version is treated as the fix; its parent is the buggy
version. `--discover N` will scan recent single-`.py`-file commits in a repo.

**From BugsInPy / Defects4J file pairs:** the *fixed* file is the clean
baseline, the *buggy* file the post-change source; call `make_buggy_pr(...)`
(same signature as the injected set) and drop the results into a dataset.

## 2-week plan → code map

- Days 1–2  real dataset → `make_example_dataset.py` / `build_dataset.py`
- Days 3–4  diff + context → `diff_parser.py`, `context_retriever.py` (done)
- Day 5     LLM structured call → `prompts.py`, `engines.py` (done; refine prompt)
- Days 6–7  calibration + providers → `calibration.py`, `providers.py` (tune)
- Days 8–9  ablation → `run_ablation.py` (headline result)
- Day 10    cross-provider → rerun with `--provider openai`
- Days 11–12 summaries + small human rating (the `summary` field is produced)
- Days 13–14 plots + writeup (the HTML report is a starting point)

## Honest limitations

- Static detector is intentionally pattern-based — semantic bugs are out of scope
  for it by design (that's the point of the comparison).
- Context retrieval is single-file (enclosing defs + same-file symbols).
- Verification (linter + tests) runs under resource limits, not a full jail;
  containerise before running untrusted PRs.
- The agent variant is deliberately omitted (it hurts reproducibility); future work.
