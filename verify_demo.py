"""Tool-augmented verification demo: a failing test catches a semantic bug
(off-by-one) that the static analyser cannot, and is reported as VERIFIED.

    python verify_demo.py
"""
from pullpilot.benchmark.build_dataset import load_dataset
from pullpilot.engines import StaticAnalysisEngine
from pullpilot.reviewer import PullRequest, Reviewer

prs = load_dataset("data/examples/dataset.json")
bug = next(p for p in prs if p.id == "b01")  # off-by-one: return items[len(items)]

# A test that exercises the changed function.
TEST = """
from indexing import get_last

def test_get_last():
    assert get_last([10, 20, 30]) == 30
"""

pr = PullRequest(diff=bug.diff, post_files={bug.file: bug.post_source},
                 title=bug.title, tests=TEST)

print("=== diff ===")
print(bug.diff)

for verify in (False, True):
    review = Reviewer(StaticAnalysisEngine(), verify=verify).review(pr)
    label = "WITH verification (linter + tests)" if verify else "static only"
    print(f"--- {label}: {len(review.issues)} finding(s) ---")
    for i in review.sorted_issues():
        tag = "VERIFIED" if i.verified else "inferred"
        print(f"  [{tag:8}] {i.severity.value:<8} {i.source:<6} "
              f"{i.file}:{i.line_start}-{i.line_end}  {i.explanation[:70]}")
    print()
