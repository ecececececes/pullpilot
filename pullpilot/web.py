"""PullPilot web UI. Runs locally and uses the real review engines.

    pip install flask
    python -m pullpilot.web
    # open http://localhost:5000

Two pages:
  /            paste a diff (+ optional full file) -> live review
  /dashboard   benchmark numbers from data/examples/results.json
"""
from __future__ import annotations

import json
import os

from flask import Flask, jsonify, render_template_string, request

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


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PullPilot</title><style>
:root{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#16191d; --mut:#697078; --line:#e4e7eb;
  --accent:#3b5bdb; --crit:#d6336c; --major:#e8590c; --minor:#f08c00; --info:#1c7ed6;
  --verified:#2b8a3e; --add:#2b8a3e; --del:#e03131;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
header{display:flex;align-items:baseline;gap:.6rem;padding:1.1rem 1.4rem;border-bottom:1px solid var(--line);background:var(--panel)}
header .logo{font-family:var(--mono);font-weight:700;font-size:1.15rem;letter-spacing:-.02em}
header .logo b{color:var(--accent)} header .tag{color:var(--mut);font-size:.85rem}
header nav{margin-left:auto;display:flex;gap:1rem;font-size:.9rem}
header a{color:var(--mut);text-decoration:none} header a:hover{color:var(--ink)}
main{max-width:1100px;margin:1.4rem auto;padding:0 1.4rem;display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media(max-width:820px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.1rem}
.panel h2{margin:.1rem 0 .9rem;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
label{display:block;font-size:.8rem;color:var(--mut);margin:.6rem 0 .3rem}
textarea,select{width:100%;font-family:var(--mono);font-size:.82rem;border:1px solid var(--line);border-radius:8px;padding:.6rem;background:#fcfcfd;color:var(--ink)}
textarea{resize:vertical} #diff{height:150px} #source{height:170px}
select{font-family:var(--sans)}
.row{display:flex;gap:.8rem;align-items:end;flex-wrap:wrap;margin-top:.7rem}
.row>div{flex:1;min-width:130px}
.check{display:flex;align-items:center;gap:.4rem;font-size:.85rem;color:var(--ink)}
button{font-family:var(--sans);font-weight:600;font-size:.92rem;border:0;border-radius:8px;padding:.62rem 1.1rem;background:var(--accent);color:#fff;cursor:pointer}
button:hover{filter:brightness(1.05)} button:disabled{opacity:.5;cursor:default}
.summary{font-size:.95rem;background:#f1f3ff;border:1px solid #d7ddff;border-radius:8px;padding:.7rem .85rem;margin-bottom:.9rem}
.empty{color:var(--mut);font-size:.9rem;text-align:center;padding:2rem 0}
.issue{border:1px solid var(--line);border-left:4px solid var(--mut);border-radius:8px;padding:.65rem .8rem;margin-bottom:.6rem}
.issue.critical{border-left-color:var(--crit)} .issue.major{border-left-color:var(--major)}
.issue.minor{border-left-color:var(--minor)} .issue.informational{border-left-color:var(--info)}
.issue .top{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;font-size:.74rem;margin-bottom:.3rem}
.pill{padding:.08rem .45rem;border-radius:999px;font-weight:600}
.pill.sev{color:#fff} .sev.critical{background:var(--crit)} .sev.major{background:var(--major)}
.sev.minor{background:var(--minor)} .sev.informational{background:var(--info)}
.pill.loc{font-family:var(--mono);background:#eef1f4;color:var(--ink)}
.pill.v{background:#e6f4ea;color:var(--verified)} .pill.i{background:#f1f3f5;color:var(--mut)}
.pill.t{background:#f1f3f5;color:var(--mut)} .conf{margin-left:auto;color:var(--mut)}
.issue p{margin:.2rem 0 0;font-size:.9rem} .issue .fix{margin-top:.35rem;font-size:.83rem;color:var(--mut)}
.q{color:var(--info);font-weight:600}
.err{background:#fff0f3;border:1px solid #ffc9d4;color:#a61e4d;border-radius:8px;padding:.7rem .85rem;font-size:.88rem}
.hint{font-size:.78rem;color:var(--mut);margin-top:.5rem}
</style></head><body>
<header>
  <span class="logo">Pull<b>Pilot</b></span>
  <span class="tag">context-grounded PR review</span>
  <nav><a href="/">Review</a><a href="/dashboard">Benchmark</a></nav>
</header>
<main>
  <section class="panel">
    <h2>Change to review</h2>
    <label for="example">Load an example</label>
    <select id="example"><option value="">— pick a sample pull request —</option></select>
    <label for="diff">Diff</label>
    <textarea id="diff" placeholder="Paste a unified diff here..."></textarea>
    <label for="source">Full file after the change (helps context + the static engine)</label>
    <textarea id="source" placeholder="Paste the full post-change file (optional)..."></textarea>
    <div class="row">
      <div>
        <label for="engine">Engine</label>
        <select id="engine">__ENGINE_OPTIONS__</select>
      </div>
      <div class="check"><input type="checkbox" id="verify"><label for="verify" style="margin:0">Run linter check</label></div>
      <div style="flex:0"><button id="go">Review change</button></div>
    </div>
    <p class="hint">Static needs no key. Free engines (gemini, groq, github, openrouter) read their key from the matching environment variable in the terminal you launched this from.</p>
  </section>
  <section class="panel">
    <h2>Review</h2>
    <div id="out"><div class="empty">Pick a sample or paste a diff, then press Review.</div></div>
  </section>
</main>
<script>
const EXAMPLES = __EXAMPLES__;
const sel = document.getElementById('example');
EXAMPLES.forEach(e=>{const o=document.createElement('option');o.value=e.id;
  o.textContent=(e.label==='buggy'?'🐞 ':'✅ ')+e.title+' ('+e.file+')';sel.appendChild(o);});
sel.onchange=()=>{const e=EXAMPLES.find(x=>x.id===sel.value);if(!e)return;
  document.getElementById('diff').value=e.diff;document.getElementById('source').value=e.source;};
const out=document.getElementById('out'), go=document.getElementById('go');
go.onclick=async()=>{
  const body={diff:document.getElementById('diff').value,
    source:document.getElementById('source').value,
    file:(EXAMPLES.find(x=>x.id===sel.value)||{}).file||'input.py',
    engine:document.getElementById('engine').value,
    verify:document.getElementById('verify').checked};
  if(!body.diff.trim()){out.innerHTML='<div class="err">Add a diff first.</div>';return;}
  go.disabled=true;out.innerHTML='<div class="empty">Reviewing… (LLM engines can take a moment)</div>';
  try{
    const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){out.innerHTML='<div class="err">'+d.error+'</div>';go.disabled=false;return;}
    let h='<div class="summary">'+(d.summary||'(no summary)')+'</div>';
    if(!d.issues.length){h+='<div class="empty">No issues found in the changed lines.</div>';}
    d.issues.forEach(i=>{
      const badge=i.verified?'<span class="pill v">verified</span>':'<span class="pill i">inferred</span>';
      h+='<div class="issue '+i.severity+'"><div class="top">'+
        '<span class="pill sev '+i.severity+'">'+i.severity+'</span>'+
        '<span class="pill loc">'+i.file+':'+i.line_start+'-'+i.line_end+'</span>'+
        '<span class="pill t">'+i.type+'</span>'+badge+
        '<span class="conf">'+Math.round(i.confidence*100)+'%</span></div>'+
        '<p>'+(i.is_question?'<span class="q">? </span>':'')+escapeHtml(i.explanation)+'</p>'+
        (i.suggested_fix?'<div class="fix">fix: '+escapeHtml(i.suggested_fix)+'</div>':'')+'</div>';
    });
    out.innerHTML=h;
  }catch(e){out.innerHTML='<div class="err">Request failed: '+e+'</div>';}
  go.disabled=false;
};
function escapeHtml(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
</script></body></html>"""


DASH = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PullPilot · Benchmark</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#16191d;--mut:#697078;--line:#e4e7eb;--accent:#3b5bdb;--ok:#2b8a3e;--miss:#c92a2a;--warn:#e8590c}
body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:baseline;gap:.6rem;padding:1.1rem 1.4rem;border-bottom:1px solid var(--line);background:var(--panel)}
header .logo{font-family:ui-monospace,Menlo,monospace;font-weight:700;font-size:1.15rem}.logo b{color:var(--accent)}
header nav{margin-left:auto;display:flex;gap:1rem;font-size:.9rem}header a{color:var(--mut);text-decoration:none}header a:hover{color:var(--ink)}
main{max-width:900px;margin:1.4rem auto;padding:0 1.4rem}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.1rem;margin-bottom:1.2rem}
h2{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin:.1rem 0 .8rem}
table{width:100%;border-collapse:collapse;font-size:.9rem}th,td{padding:.5rem;border-bottom:1px solid var(--line);text-align:center}
th:first-child,td:first-child{text-align:left;color:var(--mut)}thead th{color:var(--ink);font-weight:700}
.ok{color:var(--ok);font-weight:600}.miss{color:var(--miss)}.warn{color:var(--warn);font-weight:600}
.empty{color:var(--mut);text-align:center;padding:2rem}.empty code{background:#eef1f4;padding:.1rem .4rem;border-radius:5px}
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
    diff = data.get("diff", "")
    if not diff.strip():
        return jsonify({"error": "No diff provided."}), 400
    engine_name = data.get("engine", "static")
    file = data.get("file") or "input.py"
    try:
        engine = (StaticAnalysisEngine() if engine_name == "static"
                  else LLMEngine(get_provider(engine_name)))
        pr = PullRequest(diff=diff, post_files={file: data.get("source", "")},
                         title=data.get("title", ""))
        review = Reviewer(engine, use_context=True,
                          verify=bool(data.get("verify"))).review(pr)
        issues = [{
            "file": i.file, "line_start": i.line_start, "line_end": i.line_end,
            "type": i.type.value, "severity": i.severity.value,
            "confidence": i.confidence, "explanation": i.explanation,
            "suggested_fix": i.suggested_fix, "is_question": i.is_question,
            "source": i.source, "verified": i.verified,
        } for i in review.sorted_issues()]
        return jsonify({"summary": review.summary, "issues": issues})
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


import os

def main():
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 5000))

    print(f"PullPilot UI → http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
