"""Review one PR and print the result.

    python demo.py                 # real static-analysis engine (offline)
    python demo.py llm anthropic   # LLM engine (needs ANTHROPIC_API_KEY)
"""
import sys

from pullpilot.benchmark.build_dataset import load_dataset
from pullpilot.engines import LLMEngine, StaticAnalysisEngine
from pullpilot.providers import get_provider
from pullpilot.reviewer import PullRequest, Reviewer

mode = sys.argv[1] if len(sys.argv) > 1 else "static"
if mode == "llm":
    engine = LLMEngine(get_provider(sys.argv[2] if len(sys.argv) > 2 else "mock"))
else:
    engine = StaticAnalysisEngine()

prs = load_dataset("data/examples/dataset.json")
pr_data = next(p for p in prs if p.id == "b02")  # mutable-default bug

reviewer = Reviewer(engine, use_context=True)
review = reviewer.review(
    PullRequest(diff=pr_data.diff, post_files={pr_data.file: pr_data.post_source},
                title=pr_data.title, description=pr_data.description)
)

print(f"\nPR: {pr_data.title}  ({pr_data.file})")
print("--- diff ---")
print(pr_data.diff)
print(f"Summary: {review.summary}\n")
if not review.issues:
    print("No issues reported.")
for i, issue in enumerate(review.sorted_issues(), 1):
    tag = "QUESTION" if issue.is_question else "ISSUE"
    print(f"[{tag}] #{i}  {issue.severity.value.upper()}  {issue.type.value}  "
          f"({issue.file}:{issue.line_start})  conf={issue.confidence:.2f}")
    print(f"   {issue.explanation}")
    if issue.suggested_fix:
        print(f"   fix: {issue.suggested_fix}")
