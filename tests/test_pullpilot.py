"""End-to-end and unit tests. Run with: pytest -q"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pullpilot.diff_parser import parse_diff
from pullpilot.context_retriever import PythonASTRetriever, NoContextRetriever
from pullpilot.static_analysis import analyze
from pullpilot.calibration import calibrate
from pullpilot.schema import Issue, IssueType, Severity, Review
from pullpilot.engines import StaticAnalysisEngine
from pullpilot.reviewer import Reviewer, PullRequest
from pullpilot.benchmark.build_dataset import (
    make_buggy_pr, make_clean_pr, _ground_truth_lines,
)
from pullpilot.benchmark.evaluate import evaluate


DIFF = """--- a/m.py
+++ b/m.py
@@ -1,2 +1,2 @@
 def f(x):
-    return x + 1
+    return x - 1
"""


def test_diff_parser_changed_lines():
    parsed = parse_diff(DIFF)
    assert len(parsed.files) == 1
    fc = parsed.files[0]
    assert fc.path == "m.py"
    assert 2 in fc.added_lines
    assert 2 in fc.affected_lines


def test_context_retriever_finds_enclosing_function():
    src = "def f(x):\n    y = helper(x)\n    return y\n\ndef helper(z):\n    return z * 2\n"
    ctx = PythonASTRetriever().retrieve(src, {2})
    assert "def f" in ctx
    assert "helper" in ctx  # referenced symbol pulled in
    assert NoContextRetriever().retrieve(src, {2}) == ""


def test_static_detects_mutable_default():
    src = "def add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n"
    issues = analyze(src, {1, 2, 3}, "x.py")
    assert any("Mutable default" in i.explanation for i in issues)


def test_static_detects_eq_none_and_bare_except():
    src = "def f(x):\n    return x == None\n"
    assert any("None" in i.explanation for i in analyze(src, {2}, "x.py"))
    src2 = "def f(s):\n    try:\n        return int(s)\n    except:\n        return 0\n"
    assert any("Bare 'except" in i.explanation for i in analyze(src2, {4}, "x.py"))


def test_static_misses_semantic_bug():
    # off-by-one is not pattern-detectable; static analyser should stay silent
    src = "def get_last(items):\n    return items[len(items)]\n"
    assert analyze(src, {2}, "x.py") == []


def test_static_no_false_alarm_on_clean_code():
    src = "def add(a: int, b: int) -> int:\n    return a + b\n"
    assert analyze(src, {1, 2}, "x.py") == []


def test_calibration_drops_ungrounded_and_lowconf():
    issues = [
        Issue(file="x.py", line_start=2, line_end=2, type=IssueType.BUG,
              severity=Severity.MAJOR, confidence=0.9, explanation="grounded"),
        Issue(file="x.py", line_start=99, line_end=99, type=IssueType.BUG,
              severity=Severity.MAJOR, confidence=0.9, explanation="ungrounded"),
        Issue(file="x.py", line_start=2, line_end=2, type=IssueType.BUG,
              severity=Severity.MINOR, confidence=0.1, explanation="too low conf"),
    ]
    out = calibrate(Review(issues=issues), {"x.py": {2}})
    kept = [i.explanation for i in out.issues]
    assert "grounded" in kept
    assert "ungrounded" not in kept
    assert "too low conf" not in kept


def test_calibration_marks_questions():
    issue = Issue(file="x.py", line_start=2, line_end=2, type=IssueType.BUG,
                  severity=Severity.MINOR, confidence=0.45, explanation="maybe")
    out = calibrate(Review(issues=[issue]), {"x.py": {2}})
    assert out.issues[0].is_question is True


def test_ground_truth_replace_and_delete():
    # replace
    assert _ground_truth_lines("a\nb\nc\n", "a\nX\nc\n") == [2]
    # deletion anchors to the surviving line
    gt = _ground_truth_lines("a\nb\nc\n", "a\nc\n")
    assert gt and gt[0] in (2,)


def test_metrics_on_known_reviews():
    buggy = make_buggy_pr("p1", "x.py",
                          "def f(x):\n    return x == None\n",
                          "def f(x):\n    return x is None\n",
                          category="eq_none")
    clean = make_clean_pr("p2", "y.py",
                          "def g(a):\n    return a\n",
                          "def g(a):\n    return a  # noop\n")
    reviewer = Reviewer(StaticAnalysisEngine())
    reviews = {
        p.id: reviewer.review(PullRequest(diff=p.diff, post_files={p.file: p.post_source}))
        for p in (buggy, clean)
    }
    m = evaluate([buggy, clean], reviews)
    assert m.n_buggy == 1 and m.n_clean == 1
    assert m.recall == 1.0           # the eq_none bug is detected
    assert m.false_alarms_per_clean_pr == 0.0


def test_end_to_end_dataset_runs():
    from pullpilot.benchmark.make_example_dataset import BUGGY
    fixed = BUGGY[1][4]   # mutable_default fixed
    buggy = BUGGY[1][5]
    pr = make_buggy_pr("t", "x.py", buggy, fixed, category="mutable_default")
    reviewer = Reviewer(StaticAnalysisEngine())
    review = reviewer.review(PullRequest(diff=pr.diff, post_files={pr.file: pr.post_source}))
    assert len(review.issues) >= 1


def test_github_loader_builds_pr_offline(monkeypatch):
    """Prove the real-data shaping without hitting the network."""
    from pullpilot.benchmark import github_loader as gl

    fixed = "def f(x):\n    if x is None:\n        return 0\n    return x[0]\n"
    buggy = "def f(x):\n    return x[0]\n"  # parent (buggy) dropped the None guard

    monkeypatch.setattr(gl, "commit_info", lambda o, r, s: {
        "parents": [{"sha": "parentsha"}],
        "commit": {"message": "Fix None handling in f\n\ndetails"},
    })
    monkeypatch.setattr(gl, "fetch_file", lambda o, r, sha, p:
                        fixed if sha != "parentsha" else buggy)

    pr = gl.build_pr_from_commit("acme", "widget", "fixsha123", "w.py")
    assert pr.label == "buggy"
    assert pr.category == "real"
    assert pr.title.startswith("Fix None handling")
    assert pr.ground_truth_lines  # the reintroduced bug is localised
    # the diff re-introduces the bug (fixed -> buggy), so it removes the guard
    assert "is None" in pr.diff


def test_comparison_report_renders(tmp_path):
    from pullpilot.benchmark.make_example_dataset import BUGGY
    from pullpilot.benchmark.build_dataset import make_buggy_pr
    from pullpilot.engines import StaticAnalysisEngine
    from pullpilot.benchmark.run_comparison import _run_engine
    from pullpilot.report import render_comparison

    prs = [make_buggy_pr(b[0], b[1], b[5], b[4], category=b[2], title=b[3])
           for b in BUGGY[:4]]
    metrics, rows = _run_engine(StaticAnalysisEngine(), prs)
    out = tmp_path / "cmp.html"
    render_comparison("t", [("static", metrics, rows)], str(out))
    html = out.read_text()
    assert "Detection by bug category" in html and "static" in html


def test_run_linter_flags_undefined_name():
    from pullpilot.verification import run_linter
    src = "def greet(name):\n    return message\n"   # 'message' undefined
    issues = run_linter(src, "g.py")
    assert any(i.source == "linter" for i in issues)
    assert all(i.verified for i in issues)


def test_run_tests_detects_failure_and_passes_clean():
    from pullpilot.verification import run_tests
    buggy = {"m.py": "def double(x):\n    return x + x + 1\n"}  # wrong
    good = {"m.py": "def double(x):\n    return x * 2\n"}
    test = "from m import double\n\ndef test_double():\n    assert double(3) == 6\n"
    fails = run_tests(buggy, test)
    assert len(fails) == 1 and fails[0].source == "test" and fails[0].verified
    assert run_tests(good, test) == []


def test_calibration_keeps_verified_even_if_ungrounded():
    from pullpilot.schema import Issue, IssueType, Severity, Review
    from pullpilot.calibration import calibrate
    v = Issue(file="x.py", line_start=999, line_end=999, type=IssueType.LOGIC,
              severity=Severity.CRITICAL, confidence=0.99, explanation="test failed",
              source="test")
    out = calibrate(Review(issues=[v]), {"x.py": {1}})  # line 999 is ungrounded
    assert len(out.issues) == 1 and out.issues[0].verified


def test_verified_findings_sort_first():
    from pullpilot.schema import Issue, IssueType, Severity, Review
    model = Issue(file="x.py", line_start=1, line_end=1, type=IssueType.BUG,
                  severity=Severity.CRITICAL, confidence=0.9, explanation="m")
    verified = Issue(file="x.py", line_start=2, line_end=2, type=IssueType.STYLE,
                     severity=Severity.MINOR, confidence=0.5, explanation="v", source="linter")
    ordered = Review(issues=[model, verified]).sorted_issues()
    assert ordered[0].verified  # verified comes first despite lower severity


def test_aggregate_report_has_real_numbers(tmp_path):
    from pullpilot.benchmark import aggregate_results as agg
    ds = os.path.join(os.path.dirname(__file__), "..", "data", "examples", "dataset.json")
    results = agg.collect(ds, providers=[], do_ablation=False)
    assert "static" in results["engines"]
    assert results["n_buggy"] == 20 and results["n_clean"] == 8
    md = agg.render_report(results)
    assert "## 5. Results" in md
    assert "| detection recall |" in md          # real metric table present
    assert "pending" in md                         # ablation marked pending (no LLM)


def test_format_review_markdown():
    from pullpilot.github_review import format_review_markdown
    from pullpilot.schema import Issue, IssueType, Severity, Review
    rev = Review(summary="Adds a helper.", issues=[
        Issue(file="a.py", line_start=3, line_end=3, type=IssueType.BUG,
              severity=Severity.CRITICAL, confidence=0.99, explanation="fails a test",
              source="test"),
        Issue(file="a.py", line_start=5, line_end=5, type=IssueType.STYLE,
              severity=Severity.MINOR, confidence=0.5, explanation="maybe rename",
              is_question=True),
    ])
    md = format_review_markdown(rev)
    assert "PullPilot review" in md
    assert "critical" in md and "verified" in md      # verified test finding shown
    assert "a.py:3-3" in md
    assert "❓" in md                                   # question marker
    assert "rests with a human" in md                  # disclaimer
    # verified/critical sorts before the inferred minor
    assert md.index("a.py:3-3") < md.index("a.py:5-5")


def test_build_pr_from_pull_offline(monkeypatch):
    from pullpilot import github_review as gr
    diff = ("--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n"
            " def f(x):\n-    return x\n+    return y\n")
    monkeypatch.setattr(gr, "get_pull", lambda o, r, n: {
        "head": {"sha": "deadbeef"}, "title": "tweak f", "body": "desc"})
    monkeypatch.setattr(gr, "get_pull_diff", lambda o, r, n: diff)
    monkeypatch.setattr(gr, "fetch_file", lambda o, r, sha, p: "def f(x):\n    return y\n")
    pr = gr.build_pr("acme", "widget", 7)
    assert pr.title == "tweak f"
    assert "m.py" in pr.post_files and "return y" in pr.post_files["m.py"]
    assert "@@" in pr.diff


def test_free_provider_presets_exist():
    from pullpilot.providers import PRESETS, get_provider
    for p in ("gemini", "groq", "github", "ollama"):
        assert p in PRESETS
    # unknown provider gives a helpful error listing presets
    try:
        get_provider("definitely-not-real")
        assert False, "should have raised"
    except ValueError as e:
        assert "gemini" in str(e)


def test_discover_bugfix_commits_offline(monkeypatch):
    from pullpilot.benchmark import github_loader as gl
    commits = [
        {"sha": "aaa", "commit": {"message": "Fix off-by-one in slicing"}},
        {"sha": "bbb", "commit": {"message": "Add new feature for charts"}},   # not a fix
        {"sha": "ccc", "commit": {"message": "fix crash when value is None"}},
    ]
    details = {
        "aaa": {"parents": [{"sha": "aaa0"}], "files": [
            {"filename": "pkg/util.py", "status": "modified", "changes": 4}]},
        "ccc": {"parents": [{"sha": "ccc0"}], "files": [
            {"filename": "pkg/core.py", "status": "modified", "changes": 6},
            {"filename": "tests/test_core.py", "status": "modified", "changes": 10}]},
    }
    monkeypatch.setattr(gl, "list_commits", lambda o, r, per_page=100: commits)
    monkeypatch.setattr(gl, "commit_info", lambda o, r, s: details[s])
    pairs = gl.discover_bugfix_commits("acme", "widget", n=5)
    shas = [p[0] for p in pairs]
    assert "aaa" in shas and "ccc" in shas      # both bug fixes found
    assert "bbb" not in shas                     # feature commit skipped
    # the test file is excluded; core.py is chosen for ccc
    assert ("ccc", "pkg/core.py") in pairs


def test_web_endpoints():
    from pullpilot import web
    client = web.app.test_client()
    # index renders and includes the engine dropdown + examples
    r = client.get("/")
    assert r.status_code == 200 and b"PullPilot" in r.data and b"static" in r.data
    # review API: static engine on the mutable-default bug
    payload = {
        "diff": "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-def f(a=None):\n+def f(a=[]):\n     return a\n",
        "source": "def f(a=[]):\n    return a\n",
        "file": "x.py", "engine": "static", "verify": False,
    }
    r = client.post("/api/review", json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert "issues" in data
    assert any("Mutable default" in i["explanation"] for i in data["issues"])
    # dashboard renders (with or without results.json)
    assert client.get("/dashboard").status_code == 200
    # empty diff -> error
    assert client.post("/api/review", json={"diff": ""}).status_code == 400


def test_verification_dedupes_model_and_linter_duplicate():
    # The static engine's own pyflakes pass and the separate linter-verification
    # pass can report the identical undefined-name finding under different
    # `source` values; the reviewer must collapse them into one, keeping the
    # verified (linter) one rather than showing both.
    diff = "--- a/y.py\n+++ b/y.py\n@@ -1,2 +1,2 @@\n-def g():\n+def g():\n+    return undefined_var\n"
    source = "def g():\n    return undefined_var\n"
    pr = PullRequest(diff=diff, post_files={"y.py": source}, title="", description="")
    review = Reviewer(StaticAnalysisEngine(), use_context=True, verify=True).review(pr)
    matches = [i for i in review.issues if "undefined name 'undefined_var'" in i.explanation]
    assert len(matches) == 1
    assert matches[0].source == "linter" and matches[0].verified


def test_fetch_github_pr_reconstructs_diff_headers_and_fetches_files(monkeypatch):
    import base64
    from pullpilot import web

    src_a = "def f(a=[]):\n    return a\n"
    src_b = "y = 1\n"

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_get(url, headers=None, timeout=None, params=None):
        if url.endswith("/pulls/42"):
            return FakeResp({
                "title": "Fix mutable default",
                "body": "desc",
                "base": {"ref": "main"},
                "head": {"ref": "fix-branch", "sha": "deadbeef"},
            })
        if url.endswith("/pulls/42/files"):
            return FakeResp([
                {"filename": "a.py", "status": "modified",
                 "patch": "@@ -1,2 +1,2 @@\n-def f(a=None):\n+def f(a=[]):\n     return a"},
                {"filename": "b.py", "status": "added", "patch": "@@ -0,0 +1 @@\n+y = 1"},
            ])
        if "/contents/a.py" in url:
            return FakeResp({"encoding": "base64",
                              "content": base64.b64encode(src_a.encode()).decode()})
        if "/contents/b.py" in url:
            return FakeResp({"encoding": "base64",
                              "content": base64.b64encode(src_b.encode()).decode()})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(web.requests, "get", fake_get)
    result = web._fetch_github_pr("https://github.com/acme/widget/pull/42")

    assert result["base_ref"] == "main" and result["head_ref"] == "fix-branch"
    assert result["post_files"] == {"a.py": src_a, "b.py": src_b}
    # must be a valid unified diff (real per-file headers reconstructed)
    parsed = parse_diff(result["diff"])
    assert {fc.path for fc in parsed.files} == {"a.py", "b.py"}
