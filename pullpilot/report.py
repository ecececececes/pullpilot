"""Render benchmark results to a standalone HTML report."""
from __future__ import annotations

import html
from typing import List


def _esc(s) -> str:
    return html.escape(str(s))


def render_comparison(title: str, results, out_path: str) -> str:
    """results: list of (engine_name, metrics, rows)."""
    engines = [r[0] for r in results]

    metric_specs = [
        ("detection recall", lambda m: f"{m.recall:.0%}"),
        ("precision", lambda m: f"{m.precision:.0%}"),
        ("exact localization", lambda m: f"{m.exact_localization:.0%}"),
        ("false alarms / clean PR", lambda m: f"{m.false_alarms_per_clean_pr:.2f}"),
    ]
    head = "".join(f"<th>{_esc(e)}</th>" for e in engines)
    metric_rows = ""
    for label, fmt in metric_specs:
        cells = "".join(f"<td>{fmt(m)}</td>" for _, m, _ in results)
        metric_rows += f"<tr><th>{label}</th>{cells}</tr>"

    # per-category detection matrix
    cats = sorted({r["category"] for _, _, rows in results for r in rows
                   if r["label"] == "buggy" and r["category"]})
    det = {e: {} for e in engines}
    for name, _, rows in results:
        for r in rows:
            if r["label"] == "buggy" and r["category"]:
                cell = det[name].setdefault(r["category"], [0, 0])
                cell[1] += 1
                if r["detected"]:
                    cell[0] += 1
    cat_rows = ""
    for cat in cats:
        cells = ""
        for e in engines:
            hit, tot = det[e].get(cat, [0, 0])
            cls = "ok" if hit == tot and tot else ("miss" if hit == 0 else "warn")
            cells += f'<td class="{cls}">{hit}/{tot}</td>'
        cat_rows += f"<tr><th>{_esc(cat)}</th>{cells}</tr>"

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{_esc(title)}</title><style>
:root{{--ink:#1a1a2e;--mut:#6b7280;--ok:#0a7d33;--miss:#b00020;--warn:#b06a00;--line:#e5e7eb}}
body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);max-width:900px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{margin-top:2rem;font-size:1.1rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}}
table{{width:100%;border-collapse:collapse;font-size:.9rem;margin-top:.6rem}}
th,td{{text-align:center;padding:.5rem;border-bottom:1px solid var(--line)}}
th:first-child,td:first-child{{text-align:left;color:var(--mut)}}
thead th{{color:var(--ink);font-weight:700}}
.ok{{color:var(--ok);font-weight:600}} .miss{{color:var(--miss)}} .warn{{color:var(--warn);font-weight:600}}
.note{{color:var(--mut);font-size:.85rem}}
</style></head><body>
<h1>{_esc(title)}</h1>
<h2>Metrics</h2>
<table><thead><tr><th>metric</th>{head}</tr></thead><tbody>{metric_rows}</tbody></table>
<h2>Detection by bug category</h2>
<table><thead><tr><th>category</th>{head}</tr></thead><tbody>{cat_rows}</tbody></table>
<p class="note">Each engine reviewed the same PRs. Static analysis is expected to
catch pattern bugs and miss semantic ones; an LLM engine should close that gap.</p>
</body></html>"""
    with open(out_path, "w") as f:
        f.write(doc)
    return out_path


def render_report(title: str, metrics, rows: List[dict], out_path: str) -> str:
    detected = [r for r in rows if r["label"] == "buggy"]
    clean = [r for r in rows if r["label"] == "clean"]

    metric_cards = "".join(
        f'<div class="card"><div class="num">{val}</div><div class="lbl">{lbl}</div></div>'
        for lbl, val in [
            ("detection recall", f"{metrics.recall:.0%}"),
            ("precision", f"{metrics.precision:.0%}"),
            ("exact localization", f"{metrics.exact_localization:.0%}"),
            ("false alarms / clean PR", f"{metrics.false_alarms_per_clean_pr:.2f}"),
            ("buggy PRs", metrics.n_buggy),
            ("clean PRs", metrics.n_clean),
        ]
    )

    def status(r):
        if r["label"] == "clean":
            return ('<span class="warn">FALSE ALARM</span>' if r["n_issues"]
                    else '<span class="ok">clean</span>')
        return ('<span class="ok">DETECTED</span>' if r["detected"]
                else '<span class="miss">missed</span>')

    def issue_list(r):
        if not r["issues"]:
            return "<em>none</em>"
        return "<br>".join(
            f'L{i.line_start} · {_esc(i.severity.value)} · {_esc(i.type.value)} · '
            f'{_esc(i.explanation[:90])}' for i in r["issues"]
        )

    buggy_rows = "".join(
        f"<tr><td>{_esc(r['id'])}</td><td>{_esc(r['category'])}</td>"
        f"<td>{status(r)}</td><td>{_esc(r['ground_truth'])}</td>"
        f"<td class='issues'>{issue_list(r)}</td></tr>"
        for r in detected
    )
    clean_rows = "".join(
        f"<tr><td>{_esc(r['id'])}</td><td>{_esc(r['title'])}</td>"
        f"<td>{status(r)}</td><td class='issues'>{issue_list(r)}</td></tr>"
        for r in clean
    )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{_esc(title)}</title><style>
:root{{--ink:#1a1a2e;--mut:#6b7280;--ok:#0a7d33;--miss:#b00020;--warn:#b06a00;--line:#e5e7eb}}
body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);max-width:960px;margin:2rem auto;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{margin-top:2rem;font-size:1.1rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}}
.cards{{display:flex;flex-wrap:wrap;gap:.75rem;margin:1rem 0}}
.card{{flex:1;min-width:120px;border:1px solid var(--line);border-radius:10px;padding:.8rem 1rem;text-align:center}}
.num{{font-size:1.6rem;font-weight:700}} .lbl{{color:var(--mut);font-size:.8rem;margin-top:.2rem}}
table{{width:100%;border-collapse:collapse;font-size:.86rem}}
th,td{{text-align:left;padding:.45rem .5rem;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--mut);font-weight:600}} .issues{{color:var(--mut)}}
.ok{{color:var(--ok);font-weight:600}} .miss{{color:var(--miss);font-weight:600}} .warn{{color:var(--warn);font-weight:600}}
.note{{color:var(--mut);font-size:.85rem}}
</style></head><body>
<h1>{_esc(title)}</h1>
<div class="cards">{metric_cards}</div>
<p class="note">Recall = buggy PRs with at least one finding on the planted line.
Precision = findings that landed on a real defect. False alarms = findings raised on clean PRs.</p>
<h2>Buggy PRs ({len(detected)})</h2>
<table><tr><th>id</th><th>category</th><th>result</th><th>defect line(s)</th><th>findings</th></tr>{buggy_rows}</table>
<h2>Clean PRs ({len(clean)})</h2>
<table><tr><th>id</th><th>change</th><th>result</th><th>findings</th></tr>{clean_rows}</table>
</body></html>"""

    with open(out_path, "w") as f:
        f.write(doc)
    return out_path
