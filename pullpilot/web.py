"""PullPilot web UI. Runs locally and uses the real review engines.

    pip install flask requests
    python -m pullpilot.web
    # open http://localhost:5000

Three pages:
  /            paste a diff (+ optional full file) -> live review
               OR paste a GitHub PR link -> fetch & review
  /dashboard   benchmark numbers from data/examples/results.json
"""
from __future__ import annotations

import base64
import json
import os
import re

import requests
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context

from . import history
from .diff_parser import parse_diff
from .engines import LLMEngine, StaticAnalysisEngine
from .providers import PRESETS, get_provider
from .reviewer import PullRequest, Reviewer

app = Flask(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATASET = os.path.join(_ROOT, "data", "examples", "dataset.json")
_RESULTS = os.path.join(_ROOT, "data", "examples", "results.json")

_ENGINE_CHOICES = ["static"] + sorted(PRESETS) + ["selfhosted","openai", "anthropic"]

# Point at a GitHub Enterprise Server instance (e.g. https://ghe.corp/api/v3)
# to keep PR fetches on your own network.
_GH_API = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")


def _load_examples():
    try:
        with open(_DATASET) as f:
            data = json.load(f)
    except Exception:
        return []
    out = []
    for p in data:
        out.append({"id": p["id"], "title": p.get("title") or p["id"],
                    "label": p["label"], "file": p["file"],
                    "diff": p["diff"], "source": p["post_source"]})
    return out


def _fetch_file_at_ref(owner: str, repo: str, path: str, ref: str, headers: dict) -> str | None:
    """Fetch a single file's full text content at a given ref. Returns None on
    any failure (binary file, missing file, rate limit, ...) so callers can
    degrade gracefully instead of failing the whole review."""
    try:
        resp = requests.get(
            f"{_GH_API}/repos/{owner}/{repo}/contents/{path}",
            headers=headers, params={"ref": ref}, timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("encoding") == "base64" and "content" in body:
            return base64.b64decode(body["content"]).decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def _fetch_github_pr(url: str) -> dict:
    """
    Fetch a GitHub PR by URL and extract a reviewable diff plus the full
    post-change source of every changed file.

    Accepts URLs like:
    - https://github.com/owner/repo/pull/123
    - https://api.github.com/repos/owner/repo/pulls/123

    Returns: {"diff": "...", "title": "...", "description": "...",
              "files": [...], "post_files": {path: source}, ...}
    """
    # Parse GitHub PR URL
    match = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if not match and _GH_API != "https://api.github.com":
        # A custom GITHUB_API_URL is set (self-hosted GitHub Enterprise):
        # also accept PR links on that host.
        match = re.search(r"([^/\s]+)/([^/\s]+)/pull/(\d+)", url)
    if not match:
        raise ValueError(
            "Invalid GitHub PR URL. Expected: https://github.com/owner/repo/pull/123"
        )

    owner, repo, pr_num = match.groups()
    api_url = f"{_GH_API}/repos/{owner}/{repo}/pulls/{pr_num}"

    # Use GitHub token if available (for higher rate limits)
    headers = {}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"token {token}"

    try:
        # Fetch PR details
        resp = requests.get(api_url, headers=headers, timeout=10)
        resp.raise_for_status()
        pr_data = resp.json()

        # Fetch the per-file patches
        files_resp = requests.get(
            f"{api_url}/files",
            headers=headers,
            timeout=10,
            params={"per_page": 100}
        )
        files_resp.raise_for_status()
        files_data = files_resp.json()

        head_sha = (pr_data.get("head") or {}).get("sha")

        # GitHub's per-file "patch" field is just the hunk body — it does NOT
        # include the "--- a/file" / "+++ b/file" headers a unified diff needs.
        # Concatenating raw patches without reconstructing those headers
        # produces an unparseable diff, so rebuild them here.
        diff_parts = []
        changed_files = []
        post_files: dict[str, str] = {}
        for f in files_data:
            filename = f["filename"]
            status = f.get("status")
            changed_files.append(filename)

            patch = f.get("patch")
            if patch:
                old_name = f.get("previous_filename", filename)
                old_path = "/dev/null" if status == "added" else f"a/{old_name}"
                new_path = "/dev/null" if status == "removed" else f"b/{filename}"
                diff_parts.append(f"--- {old_path}\n+++ {new_path}\n{patch}\n")

            if status != "removed" and head_sha:
                content = _fetch_file_at_ref(owner, repo, filename, head_sha, headers)
                if content is not None:
                    post_files[filename] = content

        diff_text = "".join(diff_parts)
        if not diff_text:
            raise ValueError("No diff found in this PR (may be empty or binary files only)")

        return {
            "diff": diff_text,
            "title": pr_data.get("title", f"PR #{pr_num}"),
            "description": pr_data.get("body", ""),
            "files": changed_files,
            "post_files": post_files,
            "url": url,
            "base_ref": (pr_data.get("base") or {}).get("ref"),
            "head_ref": (pr_data.get("head") or {}).get("ref"),
        }
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch PR from GitHub: {e}")


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PullPilot</title><style>
:root{
  --bg:#050810; --bg-alt:#0a0f1a; --panel:#111927; --panel-2:#0c1220;
  --line:#1e2836; --ink:#e7ebf3; --mut:#7c8798; --mut-2:#4d5866;
  --accent:#3f9dff; --accent-ink:#04101f;
  --crit:#f5416c; --major:#f6923d; --minor:#f0c419; --style:#b285f5; --info:#3f9dff;
  --good:#33c17c; --add:#33c17c; --del:#f5416c;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
header{display:flex;align-items:center;gap:.7rem;padding:1.1rem 1.4rem;border-bottom:1px solid var(--line);background:var(--bg-alt)}
header .logo{font-family:var(--mono);font-weight:700;font-size:1.1rem;letter-spacing:-.02em;color:var(--ink)}
header .logo b{color:var(--accent)}
header .tag{color:var(--accent);font-size:.88rem;font-weight:600}
header .branch-pill{display:none;font-family:var(--mono);font-size:.74rem;color:var(--mut);background:var(--panel-2);border:1px solid var(--line);border-radius:999px;padding:.25rem .7rem}
header nav{margin-left:auto;display:flex;gap:1.1rem;font-size:.85rem}
header a{color:var(--mut);text-decoration:none} header a:hover{color:var(--ink)}

.hero{max-width:1100px;margin:1.6rem auto 0;padding:0 1.4rem}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.2rem 1.3rem}
.panel h2{margin:0 0 .9rem;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--mut)}

.tabs{display:flex;gap:.35rem;margin-bottom:1rem}
.tab{padding:.4rem .85rem;cursor:pointer;border:1px solid var(--line);border-radius:999px;background:var(--panel-2);font-size:.78rem;color:var(--mut);font-weight:600}
.tab:hover{color:var(--ink)} .tab.active{color:var(--accent-ink);background:var(--accent);border-color:var(--accent)}

.field-group{display:none;flex:1;min-width:240px}
.field-group.active{display:block}
.input-row{display:flex;gap:.8rem;align-items:end;flex-wrap:wrap}
label{display:block;font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:0 0 .4rem}
textarea,select,input{width:100%;font-family:var(--mono);font-size:.82rem;border:1px solid var(--line);border-radius:8px;padding:.6rem .7rem;background:var(--panel-2);color:var(--ink)}
textarea{resize:vertical;font-family:var(--mono)} input[type="text"]{font-family:var(--mono)}
#diff{height:110px} #source{height:110px}
.field-group#paste label:nth-of-type(2){margin-top:.7rem}
select{font-family:var(--sans)}
textarea::placeholder,input::placeholder{color:var(--mut-2)}
#go{flex:0 0 auto;font-family:var(--sans);font-weight:700;font-size:.9rem;border:0;border-radius:9px;padding:.68rem 1.2rem;background:var(--accent);color:var(--accent-ink);cursor:pointer;white-space:nowrap}
#go:hover{filter:brightness(1.08)} #go:disabled{opacity:.5;cursor:default}
.settings-row{display:flex;gap:1.2rem;align-items:end;flex-wrap:wrap;margin-top:1rem;padding-top:1rem;border-top:1px solid var(--line)}
.settings-row>div{min-width:160px}
.check{display:flex;align-items:center;gap:.4rem;font-size:.82rem;color:var(--ink)}
.check input{width:auto}
.hint{font-size:.76rem;color:var(--mut-2);margin:.8rem 0 0}

main.results{max-width:1100px;margin:1.3rem auto 3rem;padding:0 1.4rem;display:grid;grid-template-columns:320px 1fr;gap:1.1rem;align-items:start}
@media(max-width:860px){main.results{grid-template-columns:1fr}}
.summary-text{font-size:.9rem;color:var(--ink);line-height:1.6}
.grounding{margin-top:1.1rem;padding-top:1rem;border-top:1px solid var(--line)}
.grounding h3{margin:0 0 .6rem;font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
.grounding-checks{display:flex;gap:.5rem;flex-wrap:wrap}
.check-pill{font-size:.74rem;font-family:var(--mono);border:1px solid var(--line);border-radius:999px;padding:.28rem .7rem;color:var(--mut-2)}
.check-pill.ok{color:var(--good);border-color:rgba(51,193,124,.4);background:rgba(51,193,124,.08)}
.check-pill.ok::before{content:"\\2713 ";}
.check-pill.off::before{content:"\\25CB ";}

.findings-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:.85rem}
.empty{grid-column:1/-1;color:var(--mut);font-size:.9rem;text-align:center;padding:2rem 0}
.err{grid-column:1/-1;background:rgba(245,65,108,.1);border:1px solid rgba(245,65,108,.35);color:#ff9db0;border-radius:8px;padding:.7rem .85rem;font-size:.88rem}

.card{background:var(--panel-2);border:1px solid var(--line);border-left:3px solid var(--mut-2);border-radius:10px;padding:.85rem .95rem;display:flex;flex-direction:column;gap:.55rem}
.card.sev-critical{border-left-color:var(--crit)} .card.sev-major{border-left-color:var(--major)}
.card.sev-minor{border-left-color:var(--minor)} .card.sev-informational{border-left-color:var(--info)}
.card-top{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap}
.pill{padding:.14rem .55rem;border-radius:999px;font-size:.68rem;font-weight:700;letter-spacing:.03em;text-transform:uppercase}
.pill.sev{color:#fff} .sev.critical{background:var(--crit)} .sev.major{background:var(--major)}
.sev.minor{background:var(--minor);color:#3a2e00} .sev.informational{background:var(--info)}
.pill.prov{margin-left:auto;font-family:var(--sans);text-transform:none;font-weight:600;font-size:.72rem}
.prov.verified{color:var(--good);background:rgba(51,193,124,.12);border:1px solid rgba(51,193,124,.35)}
.prov.inferred{color:var(--mut);background:rgba(124,135,152,.12);border:1px solid var(--line)}
.prov.low{color:var(--major);background:rgba(246,146,61,.12);border:1px solid rgba(246,146,61,.35)}
.card-loc{font-family:var(--mono);font-size:.74rem;color:var(--mut-2);display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.pill.type{background:var(--panel);border:1px solid var(--line);color:var(--mut);font-weight:600}
.card-desc{margin:0;font-size:.86rem;color:var(--ink)}
.q{color:var(--info);font-weight:700}
.diff-block{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:.5rem .65rem;font-family:var(--mono);font-size:.76rem;overflow-x:auto;white-space:pre;margin:0}
.diff-block .add{color:var(--add)} .diff-block .del{color:var(--del)}
.fix-line{font-size:.8rem;color:var(--mut);border-left:2px solid var(--line);padding-left:.55rem}
.fix-line b{color:var(--ink)}
.conf-row{margin-top:auto}
.conf-label{display:flex;justify-content:space-between;font-size:.64rem;letter-spacing:.06em;text-transform:uppercase;color:var(--mut-2)}
.conf-label .conf-pct{color:var(--ink);font-weight:700;letter-spacing:0;text-transform:none}
.conf-bar{height:5px;background:var(--line);border-radius:999px;overflow:hidden;margin-top:.3rem}
.conf-fill{height:100%;background:var(--accent);border-radius:999px}
.card-verified{font-size:.72rem;color:var(--good)}

.history-wrap{max-width:1100px;margin:0 auto 3rem;padding:0 1.4rem}
.history-list{display:flex;flex-direction:column;gap:.4rem}
.hist-row{display:flex;gap:.8rem;align-items:center;flex-wrap:wrap;font-size:.82rem;padding:.5rem .75rem;border:1px solid var(--line);border-radius:8px;background:var(--panel-2);cursor:pointer}
.hist-row:hover{border-color:var(--accent)}
.hist-time{font-family:var(--mono);font-size:.72rem;color:var(--mut-2);white-space:nowrap}
.hist-target{font-family:var(--mono);color:var(--ink);word-break:break-all}
.hist-count{margin-left:auto;color:var(--mut);white-space:nowrap}
</style></head><body>
<header>
  <div class="brand">
    <!-- Paste the SVG contents here -->
    <svg class="logo-icon" xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="20 20 330 330">
      <g>
      <path d="M 162.50 345.94 C95.78,335.65 42.58,285.63 28.03,219.50 C17.15,170.02 29.95,119.04 63.25,79.20 C84.95,53.23 117.94,33.31 151.06,26.15 C180.22,19.85 211.65,22.11 239.50,32.51 C315.98,61.07 359.14,142.41 340.31,222.50 C325.94,283.64 278.62,330.14 216.50,344.17 C206.23,346.49 173.09,347.58 162.50,345.94 ZM 216.00 338.07 C234.73,333.41 250.08,327.26 263.50,319.03 C272.45,313.55 285.11,303.62 291.43,297.12 C296.18,292.25 296.31,291.96 294.96,289.42 C291.00,282.03 277.13,272.79 263.00,268.14 C256.79,266.10 235.73,261.03 233.38,261.01 C232.77,261.00 231.00,262.91 229.46,265.24 C225.35,271.46 218.39,277.54 210.90,281.48 C189.15,292.92 162.77,289.07 146.00,272.02 C142.43,268.39 138.81,264.36 137.96,263.07 C136.79,261.30 135.88,260.90 134.21,261.43 C132.14,262.09 132.00,262.72 132.00,271.63 C132.00,284.28 129.98,287.94 123.00,287.94 C116.34,287.94 114.00,284.20 114.00,273.55 L 114.00 265.96 L 111.25 266.61 C109.74,266.97 105.11,268.55 100.96,270.12 C88.61,274.81 79.40,281.42 74.74,288.93 L 72.61 292.37 L 81.05 300.30 C102.61,320.55 131.22,334.52 161.50,339.59 C173.87,341.66 205.06,340.79 216.00,338.07 ZM 71.91 283.45 C77.22,275.65 89.79,267.85 105.00,262.91 L 113.50 260.16 L 113.44 245.83 C113.41,237.95 112.89,229.70 112.29,227.50 C111.70,225.30 110.93,221.88 110.59,219.90 C110.17,217.45 109.21,215.96 107.61,215.23 C103.73,213.46 98.01,207.94 96.02,204.04 C94.99,202.03 93.40,195.17 92.48,188.79 C90.95,178.20 90.94,176.73 92.36,171.85 C93.36,168.41 95.25,165.13 97.64,162.68 C101.35,158.87 101.37,158.82 102.71,145.78 C105.71,116.50 117.72,96.03 140.39,81.53 C153.93,72.86 167.51,69.01 184.50,69.01 C229.24,69.01 263.76,101.95 266.65,147.37 C267.27,157.12 267.42,157.65 270.75,161.36 C275.08,166.22 277.95,172.44 277.98,177.07 C278.02,184.32 274.88,201.46 272.92,204.62 C270.52,208.50 263.96,214.56 261.20,215.44 C259.65,215.93 259.01,217.30 258.46,221.27 C258.07,224.12 257.13,228.21 256.37,230.36 C255.62,232.50 255.00,237.48 255.00,241.43 C255.00,245.67 254.51,249.09 253.80,249.80 C252.23,251.37 240.77,251.37 239.20,249.80 C238.54,249.14 238.00,246.27 238.00,243.42 C238.00,237.96 236.30,235.29 233.51,236.36 C232.32,236.82 232.00,238.83 232.00,245.95 C232.00,254.49 232.12,254.99 234.25,255.43 C264.49,261.77 277.11,266.34 288.82,275.16 C291.75,277.37 295.35,280.92 296.82,283.05 C298.29,285.18 299.98,286.94 300.56,286.96 C301.92,287.01 312.34,272.72 317.85,263.27 C343.96,218.47 345.64,162.10 322.25,115.50 C299.95,71.07 259.94,40.93 210.50,31.33 C196.81,28.68 169.86,28.95 156.00,31.89 C113.61,40.89 80.20,63.42 56.44,99.04 C41.74,121.07 33.64,144.43 30.87,172.77 C28.30,198.98 34.19,229.48 46.57,254.14 C53.67,268.30 66.12,286.93 68.50,286.97 C69.05,286.98 70.58,285.40 71.91,283.45 ZM 124.42 282.39 C125.79,281.87 126.00,279.03 126.00,261.01 L 126.00 240.23 L 122.91 239.77 C121.03,239.50 119.63,239.80 119.35,240.53 C119.10,241.19 119.04,250.60 119.22,261.42 C119.50,277.39 119.84,281.29 121.03,282.04 C122.80,283.16 122.50,283.13 124.42,282.39 ZM 193.00 281.97 C204.87,280.12 219.67,271.12 224.99,262.51 C226.80,259.59 227.01,258.20 226.42,253.04 L 225.74 246.98 L 217.94 254.44 C209.06,262.92 204.87,265.35 195.27,267.61 C183.29,270.43 168.38,267.80 158.77,261.18 C156.42,259.56 152.09,255.68 149.14,252.56 C146.19,249.44 143.67,247.02 143.53,247.19 C143.39,247.36 143.02,250.06 142.72,253.20 C142.21,258.32 142.45,259.34 145.09,263.32 C150.58,271.62 162.17,278.87 174.06,281.44 C182.04,283.17 184.79,283.25 193.00,281.97 ZM 202.50 259.55 C213.46,254.19 229.93,234.80 235.01,221.28 C237.17,215.53 242.00,183.05 242.00,174.29 C242.00,165.93 236.69,158.65 229.36,156.94 C227.24,156.45 219.51,156.04 212.18,156.02 C199.47,156.00 198.64,156.13 194.22,158.71 C187.90,162.42 181.21,162.42 174.91,158.70 C170.51,156.11 169.77,156.00 156.41,156.04 C137.76,156.10 133.19,157.79 128.89,166.21 C126.44,171.01 126.50,173.55 129.54,196.28 C133.00,222.22 135.26,227.94 147.95,242.96 C156.70,253.32 161.42,257.13 169.94,260.72 C174.79,262.76 176.84,263.03 186.00,262.77 C195.25,262.52 197.21,262.13 202.50,259.55 ZM 136.34 255.25 C136.59,254.84 137.10,251.12 137.47,247.00 C138.00,241.16 137.82,239.07 136.68,237.54 C135.44,235.87 134.98,235.77 133.61,236.91 C132.35,237.95 132.00,240.19 132.00,247.12 C132.00,255.24 132.17,256.00 133.94,256.00 C135.01,256.00 136.09,255.66 136.34,255.25 ZM 250.00 242.39 C250.00,239.51 249.68,238.91 248.42,239.39 C247.55,239.73 245.97,240.00 244.92,240.00 C243.45,240.00 243.00,240.71 243.00,243.00 C243.00,245.76 243.28,246.00 246.50,246.00 C249.85,246.00 250.00,245.85 250.00,242.39 ZM 130.35 231.91 C131.65,230.06 131.60,229.35 129.93,225.57 C127.40,219.86 126.42,214.83 123.52,192.68 C121.53,177.52 121.25,172.92 122.10,168.99 C123.59,162.04 128.05,156.35 134.31,153.44 C139.17,151.17 140.61,151.02 156.86,151.01 C173.92,151.00 174.26,151.04 177.18,153.50 C179.32,155.30 181.33,156.00 184.34,156.00 C187.17,156.00 189.74,155.18 192.21,153.50 C195.76,151.09 196.46,151.00 212.08,151.00 C221.32,151.00 229.90,151.49 232.08,152.14 C239.83,154.46 246.21,162.10 247.55,170.66 C248.48,176.63 243.42,213.17 240.33,222.70 C237.16,232.51 237.80,234.29 244.31,233.80 C251.13,233.29 251.44,232.31 256.02,197.50 C258.27,180.45 260.37,162.59 260.69,157.82 L 261.29 149.14 L 255.23 145.23 C245.22,138.76 243.91,138.44 239.54,141.41 C232.30,146.32 224.72,146.05 213.06,140.46 C207.73,137.91 204.50,137.00 200.72,137.00 C196.51,137.00 195.11,136.52 192.74,134.25 C190.34,131.95 188.99,131.50 184.50,131.50 C180.01,131.50 178.66,131.95 176.26,134.25 C173.85,136.56 172.52,137.00 167.95,137.01 C164.16,137.01 161.58,137.63 159.50,139.02 C155.58,141.64 144.75,145.00 140.21,145.00 C136.46,145.00 128.00,141.48 128.00,139.91 C128.00,137.45 122.49,139.64 113.12,145.82 L 107.74 149.37 L 108.43 157.93 C108.81,162.64 111.01,180.48 113.32,197.56 C118.07,232.66 118.50,234.00 124.97,234.00 C127.81,234.00 129.29,233.43 130.35,231.91 ZM 109.03 207.75 C109.14,202.60 103.40,165.03 102.54,165.31 C100.31,166.05 97.03,173.57 97.02,178.00 C97.00,185.81 99.74,199.84 102.00,203.51 C104.86,208.12 108.97,210.61 109.03,207.75 ZM 263.03 207.98 C266.91,205.91 269.89,198.84 271.37,188.22 C273.09,175.91 272.05,169.77 267.60,166.08 C265.69,164.50 265.64,164.68 263.00,185.50 C261.88,194.30 260.71,203.19 260.40,205.25 C259.78,209.27 260.07,209.57 263.03,207.98 ZM 123.95 131.00 C123.99,125.87 122.66,125.44 116.85,128.71 C111.21,131.87 109.04,134.77 109.01,139.14 L 109.00 141.78 L 116.47 137.64 C122.64,134.22 123.94,133.07 123.95,131.00 ZM 259.92 137.25 C259.44,135.19 259.04,133.23 259.02,132.89 C258.99,132.15 246.99,125.67 246.55,126.15 C246.37,126.34 245.93,127.92 245.56,129.67 C244.90,132.73 245.10,132.96 251.69,136.88 C260.33,142.01 261.03,142.04 259.92,137.25 ZM 150.50 137.19 C164.08,131.98 176.23,116.80 173.03,109.06 C171.27,104.83 167.36,103.81 157.36,104.95 C148.31,105.99 137.85,108.47 134.82,110.30 C131.44,112.34 129.00,118.38 129.00,124.71 C129.01,138.21 136.51,142.57 150.50,137.19 ZM 232.30 138.93 C237.80,137.41 239.99,133.33 240.00,124.62 C240.00,116.55 237.83,112.09 232.73,109.69 C228.11,107.51 215.09,105.00 206.99,104.73 C200.99,104.53 199.77,104.80 197.84,106.73 C193.27,111.28 196.43,120.53 205.34,128.74 C211.45,134.38 219.31,138.24 228.00,139.89 C228.27,139.95 230.21,139.51 232.30,138.93 ZM 184.50 126.00 C187.60,126.00 191.10,126.78 193.50,128.00 C195.66,129.10 197.52,130.00 197.64,130.00 C197.76,130.00 196.32,127.45 194.44,124.33 C189.78,116.59 188.99,111.67 191.48,105.75 C192.14,104.17 191.47,104.00 184.66,104.00 L 177.10 104.00 L 178.17 107.07 C179.80,111.73 178.62,118.28 175.13,123.94 C173.41,126.74 172.00,129.20 172.00,129.41 C172.00,129.61 173.67,128.93 175.71,127.89 C177.91,126.77 181.49,126.00 184.50,126.00 ZM 118.85 121.57 C122.78,119.58 123.85,118.44 124.83,115.18 C126.81,108.56 132.25,104.58 142.63,102.14 C147.51,101.00 154.76,99.76 158.75,99.39 L 166.00 98.71 L 166.00 87.73 L 166.00 76.75 L 162.75 77.39 C157.48,78.42 143.99,85.34 137.11,90.55 C129.34,96.42 120.29,107.58 116.10,116.44 C111.88,125.37 111.77,125.16 118.85,121.57 ZM 256.00 122.90 C256.00,120.45 248.55,107.84 243.43,101.64 C238.08,95.15 226.79,85.89 220.09,82.51 C215.90,80.39 206.19,77.00 204.31,77.00 C203.21,77.00 202.99,79.14 203.22,87.69 L 203.50 98.37 L 215.00 100.02 C234.43,102.80 240.62,106.05 244.48,115.50 C245.68,118.41 247.26,120.09 250.31,121.67 C255.22,124.21 256.00,124.38 256.00,122.90 ZM 186.58 98.00 L 198.00 98.00 L 198.00 86.97 C198.00,77.40 197.77,75.84 196.25,75.22 C193.98,74.28 175.02,74.28 172.75,75.22 C171.23,75.84 171.00,77.40 171.00,86.80 C171.00,97.92 171.36,99.46 173.78,98.53 C174.54,98.24 180.30,98.00 186.58,98.00 ZM 154.97 206.41 C150.04,192.66 150.11,192.75 141.72,189.39 C137.44,187.67 133.72,186.05 133.45,185.79 C133.19,185.52 136.74,183.84 141.35,182.06 L 149.72 178.82 L 152.92 170.41 C154.68,165.78 156.28,162.00 156.48,162.00 C156.68,162.00 158.27,165.78 160.02,170.39 L 163.18 178.79 L 171.61 182.04 C176.24,183.83 179.78,185.56 179.46,185.87 C179.15,186.19 175.43,187.81 171.19,189.47 L 163.50 192.50 L 160.47 200.19 C158.81,204.43 157.12,208.21 156.73,208.61 C156.33,209.00 155.54,208.01 154.97,206.41 ZM 164.83 187.74 L 169.50 185.67 L 164.79 183.73 C160.76,182.06 159.83,181.13 158.34,177.21 L 156.59 172.64 L 154.71 177.23 C153.12,181.11 152.09,182.12 148.16,183.74 L 143.50 185.67 L 148.26 187.78 C150.88,188.95 153.23,190.48 153.48,191.20 C153.73,191.91 154.46,193.85 155.10,195.50 L 156.25 198.50 L 158.20 194.16 C159.76,190.70 161.10,189.40 164.83,187.74 ZM 208.33 198.56 C203.88,196.60 203.00,194.32 203.00,184.79 C203.00,175.21 204.15,172.77 209.47,171.01 C217.32,168.42 222.31,174.80 221.78,186.75 C221.51,192.90 221.09,194.36 218.98,196.44 C215.76,199.61 212.27,200.31 208.33,198.56 ZM 215.39 192.42 C215.73,191.55 216.00,188.21 216.00,185.00 C216.00,181.79 215.73,178.45 215.39,177.58 C214.57,175.44 210.30,175.57 209.12,177.78 C207.92,180.02 207.88,188.56 209.06,191.65 C210.10,194.39 214.43,194.92 215.39,192.42 Z" fill="rgba(255,255,255,1)"/>
      </g>
    </svg>

    
  </div>
  <span class="logo">Pull<b>Pilot</b></span>
  <span class="tag">// Code Review Copilot</span>
  <span class="branch-pill" id="branchPill"></span>
  <nav><a href="/">Review</a><a href="/dashboard">Benchmark</a></nav>
</header>

<div class="hero">
  <section class="panel">
    <div class="tabs">
      <button class="tab active" data-tab="github">GitHub PR link</button>
      <button class="tab" data-tab="paste">Paste diff</button>
      <button class="tab" data-tab="examples">Load example</button>
    </div>

    <div class="input-row">
      <div id="github" class="field-group active">
        <label for="github_url">Target PR</label>
        <input type="text" id="github_url" placeholder="github.com/owner/repo/pull/123">
      </div>

      <div id="paste" class="field-group">
        <label for="diff">Diff</label>
        <textarea id="diff" placeholder="Paste a unified diff here..."></textarea>
        <label for="source">Full file after the change (helps context + the static engine)</label>
        <textarea id="source" placeholder="Paste the full post-change file (optional)..."></textarea>
      </div>

      <div id="examples" class="field-group">
        <label for="example">Example</label>
        <select id="example"><option value="">— pick a sample pull request —</option></select>
      </div>

      <button id="go">Run Copilot Analysis</button>
    </div>

    <div class="settings-row">
      <div>
        <label for="engine">Engine</label>
        <select id="engine">__ENGINE_OPTIONS__</select>
      </div>
      <div id="keyField" style="flex:1;min-width:220px">
        <label for="api_key">API key (optional)</label>
        <input type="password" id="api_key" placeholder="paste key — else uses env var" autocomplete="off">
      </div>
      <div id="modelField" style="min-width:180px">
        <label for="model">Model (optional)</label>
        <input type="text" id="model" placeholder="engine default">
      </div>
      <div class="check"><input type="checkbox" id="verify"><label for="verify" style="margin:0">Run linter check</label></div>
    </div>
    <p class="hint">Static needs no key. For LLM engines, paste an API key above (used only for this request, never stored) or export the matching environment variable before launching. The model field overrides the engine's default (e.g. gpt-4o-mini, gemini-2.5-pro). Paste a link to any public GitHub PR to fetch and review it directly.</p>
  </section>
</div>

<main class="results">
  <section class="panel">
    <h2>Plain-Language Summary</h2>
    <div id="summaryText" class="summary-text">Pick a sample, paste a diff, or add a GitHub PR link, then press Run Copilot Analysis.</div>
    <div class="grounding">
      <h3>Grounding Context Health</h3>
      <div class="grounding-checks">
        <span class="check-pill" id="gcFile">File Mapping</span>
        <span class="check-pill" id="gcStyle">Repo Style</span>
      </div>
    </div>
  </section>

  <section class="panel">
    <h2>Structured Review Findings</h2>
    <div id="out" class="findings-grid"><div class="empty">No review yet.</div></div>
  </section>
</main>

<div class="history-wrap">
  <section class="panel">
    <h2>Recent Reviews</h2>
    <div id="history" class="history-list"><div class="empty">No reviews yet.</div></div>
  </section>
</div>

<script>
// Tabs
document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.field-group').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  };
});

const EXAMPLES = __EXAMPLES__;
const sel = document.getElementById('example');
EXAMPLES.forEach(e=>{const o=document.createElement('option');o.value=e.id;
  o.textContent=(e.label==='buggy'?'🐞 ':'✅ ')+e.title+' ('+e.file+')';sel.appendChild(o);});
sel.onchange=()=>{const e=EXAMPLES.find(x=>x.id===sel.value);if(!e)return;
  document.getElementById('diff').value=e.diff;document.getElementById('source').value=e.source;};

const out=document.getElementById('out'), go=document.getElementById('go');
const summaryEl=document.getElementById('summaryText');

// key/model only make sense for LLM engines
const engineSel=document.getElementById('engine');
function syncKeyFields(){
  const llm=engineSel.value!=='static';
  document.getElementById('keyField').style.display=llm?'':'none';
  document.getElementById('modelField').style.display=llm?'':'none';
}
engineSel.onchange=syncKeyFields; syncKeyFields();

function setGrounding(g){
  const fm=document.getElementById('gcFile'), rs=document.getElementById('gcStyle');
  fm.className='check-pill'+(g && g.file_mapping ? ' ok':' off');
  rs.className='check-pill'+(g && g.repo_style ? ' ok':' off');
}
function setBranch(b){
  const el=document.getElementById('branchPill');
  if(b && (b.base||b.head)){el.textContent=(b.base||'?')+' + '+(b.head||'?');el.style.display='inline-block';}
  else{el.style.display='none';}
}
function renderIssue(i){
  let provClass, provLabel;
  if(i.verified){provClass='verified';provLabel=i.source==='test'?'Test Verified':'Tool Matched';}
  else if(i.confidence<0.6){provClass='low';provLabel='Low Confidence';}
  else{provClass='inferred';provLabel='Inferred';}
  const typeLabel=i.type.replace(/_/g,' ').replace(/\\b\\w/g,c=>c.toUpperCase());
  const pct=Math.round(i.confidence*100);
  const loc=i.line_start===i.line_end?i.line_start:(i.line_start+'-'+i.line_end);
  let fixHtml='';
  if(i.suggested_fix){
    if(/^[-+]/m.test(i.suggested_fix)){
      const lines=i.suggested_fix.split('\\n').map(l=>{
        const cls=l.startsWith('-')?'del':(l.startsWith('+')?'add':'');
        return '<span class="'+cls+'">'+escapeHtml(l)+'</span>';
      }).join('\\n');
      fixHtml='<pre class="diff-block">'+lines+'</pre>';
    }else{
      fixHtml='<div class="fix-line"><b>Suggested fix:</b> '+escapeHtml(i.suggested_fix)+'</div>';
    }
  }
  return '<div class="card sev-'+i.severity+'"><div class="card-top">'
    +'<span class="pill sev '+i.severity+'">'+i.severity+'</span>'
    +'<span class="pill prov '+provClass+'">'+provLabel+'</span></div>'
    +'<div class="card-loc">'+i.file+':'+loc+'<span class="pill type">'+typeLabel+'</span></div>'
    +'<p class="card-desc">'+(i.is_question?'<span class="q">? </span>':'')+escapeHtml(i.explanation)+'</p>'
    +fixHtml
    +'<div class="conf-row"><div class="conf-label"><span>Engine Precision Confidence</span>'
    +'<span class="conf-pct">'+pct+'%</span></div><div class="conf-bar">'
    +'<div class="conf-fill" style="width:'+pct+'%"></div></div></div>'
    +(i.verified?'<div class="card-verified">✓ Confirmed via '+(i.source==='test'?'Test Run':'Linter Routine')+'</div>':'')
    +'</div>';
}

function renderResult(d){
  summaryEl.textContent = d.summary || '(no summary)';
  setGrounding(d.grounding);
  setBranch(d.branch);
  if(!d.issues.length){out.innerHTML='<div class="empty">No issues found in the changed lines.</div>';}
  else{out.innerHTML=d.issues.map(renderIssue).join('');}
}

async function loadHistory(){
  try{
    const r=await fetch('/api/history');
    const items=await r.json();
    const el=document.getElementById('history');
    if(!items.length){el.innerHTML='<div class="empty">No reviews yet.</div>';return;}
    el.innerHTML=items.map(h=>'<div class="hist-row" data-id="'+h.id+'">'
      +'<span class="hist-time">'+escapeHtml((h.created_at||'').replace('T',' ').replace('+00:00',''))+'</span>'
      +'<span class="hist-target">'+escapeHtml(h.target)+'</span>'
      +'<span class="pill type">'+escapeHtml(h.engine)+'</span>'
      +'<span class="hist-count">'+h.n_issues+' finding(s)</span></div>').join('');
    el.querySelectorAll('.hist-row').forEach(row=>{row.onclick=async()=>{
      const r=await fetch('/api/history/'+row.dataset.id);
      const d=await r.json();
      if(!d.error){renderResult(d);window.scrollTo({top:0,behavior:'smooth'});}
    };});
  }catch(e){}
}
loadHistory();

go.onclick=async()=>{
  const activeTab = document.querySelector('.tab.active').dataset.tab;
  let body = {
    engine: document.getElementById('engine').value,
    verify: document.getElementById('verify').checked,
    stream: true
  };
  const apiKey = document.getElementById('api_key').value.trim();
  const model = document.getElementById('model').value.trim();
  if (apiKey) body.api_key = apiKey;
  if (model) body.model = model;

  if (activeTab === 'github') {
    const url = document.getElementById('github_url').value.trim();
    if(!url){out.innerHTML='<div class="err">Add a GitHub PR URL first.</div>';return;}
    body.github_url = url;
  } else if (activeTab === 'examples') {
    const e = EXAMPLES.find(x=>x.id===sel.value);
    if(!e){out.innerHTML='<div class="err">Select an example first.</div>';return;}
    body.diff = e.diff;
    body.source = e.source;
    body.file = e.file;
  } else {
    const diff = document.getElementById('diff').value;
    if(!diff.trim()){out.innerHTML='<div class="err">Add a diff first.</div>';return;}
    body.diff = diff;
    body.source = document.getElementById('source').value;
    body.file = 'input.py';
  }

  go.disabled=true;
  out.innerHTML='<div class="empty">Starting review…</div>';
  summaryEl.textContent='Starting review…';
  const handleLine=(line)=>{
    if(!line.trim())return;
    const m=JSON.parse(line);
    if(m.stage){summaryEl.textContent=m.stage;out.innerHTML='<div class="empty">'+escapeHtml(m.stage)+'</div>';}
    else if(m.error){out.innerHTML='<div class="err">'+escapeHtml(m.error)+'</div>';summaryEl.textContent='';}
    else{renderResult(m);loadHistory();}
  };
  try{
    const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const reader=r.body.getReader(), dec=new TextDecoder();
    let buf='';
    while(true){
      const {done,value}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;
      while((i=buf.indexOf('\\n'))>=0){handleLine(buf.slice(0,i));buf=buf.slice(i+1);}
    }
    if(buf.trim())handleLine(buf);
  }catch(e){out.innerHTML='<div class="err">Request failed: '+e+'</div>';}
  go.disabled=false;
};
function escapeHtml(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
</script></body></html>"""


DASH = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PullPilot · Benchmark</title>
<style>
:root{--bg:#050810;--bg-alt:#0a0f1a;--panel:#111927;--ink:#e7ebf3;--mut:#7c8798;--line:#1e2836;
  --accent:#3f9dff;--ok:#33c17c;--miss:#f5416c;--warn:#f6923d}
body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;gap:.7rem;padding:1.1rem 1.4rem;border-bottom:1px solid var(--line);background:var(--bg-alt)}
header .logo{font-family:ui-monospace,Menlo,monospace;font-weight:700;font-size:1.1rem}.logo b{color:var(--accent)}
header nav{margin-left:auto;display:flex;gap:1.1rem;font-size:.85rem}header a{color:var(--mut);text-decoration:none}header a:hover{color:var(--ink)}
main{max-width:900px;margin:1.4rem auto;padding:0 1.4rem}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.2rem;margin-bottom:1.2rem}
h2{font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);margin:0 0 .8rem}
table{width:100%;border-collapse:collapse;font-size:.88rem}th,td{padding:.5rem;border-bottom:1px solid var(--line);text-align:center}
th:first-child,td:first-child{text-align:left;color:var(--mut)}thead th{color:var(--ink);font-weight:700}
.ok{color:var(--ok);font-weight:600}.miss{color:var(--miss)}.warn{color:var(--warn);font-weight:600}
.empty{color:var(--mut);text-align:center;padding:2rem}.empty code{background:#0c1220;border:1px solid var(--line);padding:.1rem .4rem;border-radius:5px}
.brand{
    display:flex;
    align-items:center;
    gap:.7rem;
}

.logo-icon{
    width:34px;
    height:34px;
    display:block;
    color:#fff;
}

.logo-icon *{
    fill:#fff !important;
    stroke:#fff !important;
}
</style></head><body>
<header><span class="logo">Pull<b>Pilot</b></span><nav><a href="/">Review</a><a href="/dashboard">Benchmark</a></nav></header>
<main>__BODY__</main></body></html>"""


@app.route("/")
def index():
    opts = "".join(f'<option value="{e}">{e}</option>' for e in _ENGINE_CHOICES)
    html = (PAGE.replace("__ENGINE_OPTIONS__", opts)
                .replace("__EXAMPLES__", json.dumps(_load_examples())))
    return render_template_string(html)


def _prepare_inputs(data: dict) -> dict:
    """Turn the request body (GitHub PR link or pasted diff) into review
    inputs. Raises ValueError on bad input."""
    if "github_url" in data:
        # Fetch the PR from GitHub: the real diff plus every changed file's
        # full post-change source (needed for the static engine and for
        # AST context retrieval, not just the LLM prompt).
        pr_data = _fetch_github_pr(data["github_url"])
        return {
            "diff": pr_data["diff"],
            "post_files": pr_data.get("post_files") or {},
            "title": pr_data["title"],
            "description": pr_data["description"],
            "branch": {"base": pr_data.get("base_ref"),
                       "head": pr_data.get("head_ref")},
            "target": data["github_url"],
        }
    diff = data.get("diff", "")
    if not diff.strip():
        raise ValueError("No diff provided.")
    file = data.get("file") or "input.py"
    return {"diff": diff, "post_files": {file: data.get("source", "")},
            "title": data.get("title", ""), "description": "",
            "branch": None, "target": f"pasted diff ({file})"}


def _execute_review(inp: dict, engine_name: str, verify: bool,
                    api_key: str | None, model: str | None) -> dict:
    """Run the review pipeline, persist it to history, and shape the response."""
    engine = (StaticAnalysisEngine() if engine_name == "static"
              else LLMEngine(get_provider(engine_name, api_key=api_key,
                                          model=model)))
    pr = PullRequest(diff=inp["diff"], post_files=inp["post_files"],
                     title=inp["title"], description=inp["description"])
    review = Reviewer(engine, use_context=True, verify=verify).review(pr)
    issues = [{
        "file": i.file, "line_start": i.line_start, "line_end": i.line_end,
        "type": i.type.value, "severity": i.severity.value,
        "confidence": i.confidence, "explanation": i.explanation,
        "suggested_fix": i.suggested_fix, "is_question": i.is_question,
        "source": i.source, "verified": i.verified,
    } for i in review.sorted_issues()]
    has_source = any(v.strip() for v in inp["post_files"].values())
    try:
        parse_diff(inp["diff"])
        diff_parses = True
    except Exception:
        diff_parses = False
    result = {
        "summary": review.summary, "issues": issues,
        "grounding": {"file_mapping": has_source,
                      "repo_style": has_source and diff_parses},
        "branch": inp["branch"],
    }
    try:
        history.save_review(inp["target"], engine_name, result)
    except Exception:
        pass  # history is best-effort; never fail the review over it
    return result


@app.route("/api/review", methods=["POST"])
def api_review():
    data = request.get_json(force=True) or {}
    engine_name = data.get("engine", "static")
    verify = bool(data.get("verify"))
    # Pasted key/model override the environment for this request only —
    # they are never stored or logged.
    api_key = (data.get("api_key") or "").strip() or None
    model = (data.get("model") or "").strip() or None

    if not data.get("stream"):
        try:
            inp = _prepare_inputs(data)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        try:
            return jsonify(_execute_review(inp, engine_name, verify, api_key, model))
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    # Streaming mode (the UI): NDJSON, stage lines then the final result,
    # so the user sees which slow step (GitHub fetch, model call) is running.
    def gen():
        try:
            if "github_url" in data:
                yield json.dumps({"stage": "Fetching PR from GitHub…"}) + "\n"
            inp = _prepare_inputs(data)
            stage = f"Reviewing with {engine_name}…"
            if engine_name != "static":
                stage = f"Querying {model or engine_name} (can take ~10-30s)…"
            yield json.dumps({"stage": stage}) + "\n"
            yield json.dumps(_execute_review(inp, engine_name, verify,
                                             api_key, model)) + "\n"
        except Exception as exc:
            yield json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n"
    return Response(stream_with_context(gen()),
                    mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/history")
def api_history():
    return jsonify(history.list_reviews())


@app.route("/api/history/<int:review_id>")
def api_history_item(review_id: int):
    payload = history.get_review(review_id)
    if payload is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(payload)


@app.route("/dashboard")
def dashboard():
    try:
        with open(_RESULTS) as f:
            res = json.load(f)
    except Exception:
        body = ('<div class="panel"><div class="empty">No results yet.<br><br>'
                'Run <code>python -m pullpilot.benchmark.aggregate_results --provider gemini --ablation</code>'
                ' then refresh.</div></div>')
        return render_template_string(DASH.replace("__BODY__", body))

    engines = list(res["engines"])
    # metric table
    specs = [("detection recall", "recall", "{:.0%}"), ("precision", "precision", "{:.0%}"),
             ("exact localization", "exact_localization", "{:.0%}"),
             ("false alarms / clean PR", "false_alarms_per_clean_pr", "{:.2f}")]
    head = "".join(f"<th>{e}</th>" for e in engines)
    rows = ""
    for label, key, fmt in specs:
        cells = "".join(f"<td>{fmt.format(res['engines'][e]['metrics'][key])}</td>" for e in engines)
        rows += f"<tr><td>{label}</td>{cells}</tr>"
    metric_tbl = f"<table><thead><tr><th>metric</th>{head}</tr></thead><tbody>{rows}</tbody></table>"

    # category table
    cats = res.get("categories", [])
    crows = ""
    for c in cats:
        cells = ""
        for e in engines:
            hit, tot = res["engines"][e]["by_category"].get(c, [0, 0])
            cls = "ok" if tot and hit == tot else ("miss" if hit == 0 else "warn")
            cells += f'<td class="{cls}">{hit}/{tot}</td>'
        crows += f"<tr><td>{c}</td>{cells}</tr>"
    cat_tbl = (f"<table><thead><tr><th>bug category</th>{head}</tr></thead><tbody>{crows}</tbody></table>"
               if cats else "<div class='empty'>no category data</div>")

    # ablation
    abl = ""
    for name, ab in res.get("ablation", {}).items():
        nc, wc = ab["no_context"], ab["with_context"]
        abl += f"<h2>Context ablation · {name}</h2><table><thead><tr><th>metric</th><th>no context</th><th>with context</th></tr></thead><tbody>"
        for label, key, fmt in specs:
            abl += f"<tr><td>{label}</td><td>{fmt.format(nc[key])}</td><td>{fmt.format(wc[key])}</td></tr>"
        abl += "</tbody></table>"
    abl_panel = f'<div class="panel">{abl}</div>' if abl else ""

    body = (f'<div class="panel"><h2>Engine comparison</h2>{metric_tbl}</div>'
            f'<div class="panel"><h2>Detection by bug category</h2>{cat_tbl}</div>'
            f'{abl_panel}')
    return render_template_string(DASH.replace("__BODY__", body))


def main():
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 5000))
    print(f"PullPilot UI → http://0.0.0.0:{port}  (Ctrl+C to stop)")
    print(f"selfhosted engine → "
          f"{os.environ.get('SELFHOSTED_BASE_URL', '(unset, using http://localhost:11434/api/generate)')}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
