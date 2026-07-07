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
from flask import Flask, jsonify, render_template_string, request

from .diff_parser import parse_diff
from .engines import LLMEngine, StaticAnalysisEngine
from .providers import PRESETS, get_provider
from .reviewer import PullRequest, Reviewer

app = Flask(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATASET = os.path.join(_ROOT, "data", "examples", "dataset.json")
_RESULTS = os.path.join(_ROOT, "data", "examples", "results.json")

_ENGINE_CHOICES = ["static"] + sorted(PRESETS) + ["selfhosted","openai", "anthropic"]


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
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
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
    if not match:
        raise ValueError(
            "Invalid GitHub PR URL. Expected: https://github.com/owner/repo/pull/123"
        )

    owner, repo, pr_num = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"

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
</style></head><body>
<header>
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
      <div class="check"><input type="checkbox" id="verify"><label for="verify" style="margin:0">Run linter check</label></div>
    </div>
    <p class="hint">Static needs no key. Free engines (gemini, groq, github, openrouter) read their key from the matching environment variable in the terminal you launched this from. Paste a link to any public GitHub PR to fetch and review it directly.</p>
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

go.onclick=async()=>{
  const activeTab = document.querySelector('.tab.active').dataset.tab;
  let body = {
    engine: document.getElementById('engine').value,
    verify: document.getElementById('verify').checked
  };

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
  out.innerHTML='<div class="empty">Reviewing… (LLM engines can take a moment)</div>';
  summaryEl.textContent='Running analysis…';
  try{
    const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){out.innerHTML='<div class="err">'+d.error+'</div>';summaryEl.textContent='';go.disabled=false;return;}
    summaryEl.textContent = d.summary || '(no summary)';
    setGrounding(d.grounding);
    setBranch(d.branch);
    if(!d.issues.length){out.innerHTML='<div class="empty">No issues found in the changed lines.</div>';}
    else{out.innerHTML=d.issues.map(renderIssue).join('');}
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
</style></head><body>
<header><span class="logo">Pull<b>Pilot</b></span><nav><a href="/">Review</a><a href="/dashboard">Benchmark</a></nav></header>
<main>__BODY__</main></body></html>"""


@app.route("/")
def index():
    opts = "".join(f'<option value="{e}">{e}</option>' for e in _ENGINE_CHOICES)
    html = (PAGE.replace("__ENGINE_OPTIONS__", opts)
                .replace("__EXAMPLES__", json.dumps(_load_examples())))
    return render_template_string(html)


@app.route("/api/review", methods=["POST"])
def api_review():
    data = request.get_json(force=True) or {}
    engine_name = data.get("engine", "static")
    branch = None

    # Determine if this is a GitHub PR or a direct diff
    if "github_url" in data:
        # Fetch the PR from GitHub: the real diff plus every changed file's
        # full post-change source (needed for the static engine and for
        # AST context retrieval, not just the LLM prompt).
        try:
            pr_data = _fetch_github_pr(data["github_url"])
            diff = pr_data["diff"]
            post_files = pr_data.get("post_files") or {}
            title = pr_data["title"]
            description = pr_data["description"]
            branch = {"base": pr_data.get("base_ref"), "head": pr_data.get("head_ref")}
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    else:
        # Direct diff from textarea
        diff = data.get("diff", "")
        if not diff.strip():
            return jsonify({"error": "No diff provided."}), 400
        file = data.get("file") or "input.py"
        post_files = {file: data.get("source", "")}
        title = data.get("title", "")
        description = ""

    try:
        engine = (StaticAnalysisEngine() if engine_name == "static"
                  else LLMEngine(get_provider(engine_name)))
        pr = PullRequest(diff=diff, post_files=post_files,
                         title=title, description=description)
        review = Reviewer(engine, use_context=True,
                          verify=bool(data.get("verify"))).review(pr)
        issues = [{
            "file": i.file, "line_start": i.line_start, "line_end": i.line_end,
            "type": i.type.value, "severity": i.severity.value,
            "confidence": i.confidence, "explanation": i.explanation,
            "suggested_fix": i.suggested_fix, "is_question": i.is_question,
            "source": i.source, "verified": i.verified,
        } for i in review.sorted_issues()]
        has_source = any(v.strip() for v in post_files.values())
        try:
            parse_diff(diff)
            diff_parses = True
        except Exception:
            diff_parses = False
        grounding = {
            "file_mapping": has_source,
            "repo_style": has_source and diff_parses,
        }
        return jsonify({"summary": review.summary, "issues": issues,
                        "grounding": grounding, "branch": branch})
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400


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
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
