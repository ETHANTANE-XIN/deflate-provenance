"""Court-ready report generation (HTML + JSON).

Two report kinds:

* **analysis report** for one investigated file -- the verdict, per-stream
  evidence, the deterministic signatures, and the metadata-vs-bitstream
  comparison, with the explicit caveat that the result is supporting evidence,
  not proof of authorship;
* **evaluation report** -- the confusion matrix, macro-F1, minimum-evidence
  curve, open-set behaviour and adversarial robustness, with embedded SVG
  figures.

Everything is self-contained HTML (inline CSS + inline SVG), so a report opens
in any browser with no assets.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone

from . import __version__
from .aggregate import ArchiveVerdict
from .charts import bar_svg, confusion_svg, line_svg
from .evaluate import EvalResult
from .ml.classifier import UNKNOWN

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:0;color:#1a1a1a;background:#f5f6f8}
.wrap{max-width:960px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;border-bottom:2px solid #3b7dd8;
padding-bottom:4px;margin-top:28px}
.meta{color:#666;font-size:12px}
.card{background:#fff;border:1px solid #e3e5e8;border-radius:8px;padding:16px;margin:12px 0}
.verdict{font-size:20px;font-weight:bold}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;color:#fff}
.ok{background:#2ca05a}.warn{background:#d8892b}.unknown{background:#777}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #e3e5e8;padding:6px 8px;text-align:left}
th{background:#eef2f8}
.bar{height:10px;background:#3b7dd8;border-radius:3px;display:inline-block}
.caveat{background:#fff8e1;border:1px solid #f0d78a;padding:10px;border-radius:6px;
font-size:13px}
code{background:#f0f0f3;padding:1px 4px;border-radius:3px;font-size:12px}
.small{font-size:12px;color:#555}
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _tag(verdict: ArchiveVerdict) -> str:
    if verdict.top_label in (UNKNOWN, "unknown"):
        return '<span class="tag unknown">inconclusive</span>'
    if not verdict.consistent:
        return '<span class="tag warn">mixed / rebuilt</span>'
    return '<span class="tag ok">consistent</span>'


# --- analysis report -------------------------------------------------------


def analysis_html(verdict: ArchiveVerdict, channel_compare: dict | None = None) -> str:
    rows = []
    for s in verdict.streams:
        pred = s.prediction
        label = pred.label if pred else "(no tokens)"
        conf = f"{pred.confidence:.2f}" if pred else "-"
        nov = f"{pred.novelty:.1f}" if pred else "-"
        sig = "; ".join(f"{h.rule}" for h in s.signatures) or "-"
        rows.append(
            f"<tr><td>{_esc(s.name)}</td><td>{s.out_size}</td>"
            f"<td>{s.n_blocks}</td><td>{_esc(label)}</td><td>{conf}</td>"
            f"<td>{nov}</td><td class='small'>{_esc(sig)}</td></tr>"
        )
    dist_rows = []
    for k, v in sorted(verdict.vote_distribution.items(), key=lambda kv: -kv[1]):
        dist_rows.append(
            f"<tr><td>{_esc(k)}</td><td>{v:.1%}</td>"
            f"<td><span class='bar' style='width:{int(v*200)}px'></span></td></tr>"
        )
    notes = "".join(f"<li>{_esc(n)}</li>" for n in verdict.notes) or "<li>none</li>"

    channel_html = ""
    if channel_compare:
        mc = channel_compare["metadata_channel"]
        bc = channel_compare["bitstream_channel"]
        channel_html = f"""
        <h2>Two evidence channels</h2>
        <div class="card"><table>
          <tr><th></th><th>Container metadata (forgeable)</th>
              <th>DEFLATE bitstream (this tool)</th></tr>
          <tr><td>signal</td>
              <td>host={_esc(mc.get('host_systems'))}, ver={_esc(mc.get('version_made_by'))},
                  ts={_esc(mc.get('first_timestamp'))}, extras={_esc(mc.get('extra_field_kinds'))}</td>
              <td>attribution = <b>{_esc(bc.get('attribution'))}</b></td></tr>
          <tr><td>can be forged?</td><td>yes, in seconds</td>
              <td>only by re-running the original encoder</td></tr>
        </table></div>"""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>DEFLATE provenance report</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>DEFLATE Compression-Provenance Report</h1>
<div class="meta">file: <code>{_esc(verdict.path)}</code> &middot; container: {_esc(verdict.container)}
&middot; generated {_now()} &middot; dfp v{__version__}</div>
<div class="card">
  <div class="verdict">Attribution: {_esc(verdict.top_label)} {_tag(verdict)}</div>
  <p>Most consistent with <b>{_esc(verdict.top_label)}</b>
     (vote share {verdict.top_share:.0%}, mean confidence {verdict.confidence:.0%}).
     {verdict.n_attributed} of {verdict.n_streams} streams attributed,
     {verdict.n_abstained} abstained.</p>
</div>
<h2>Vote distribution (per-archive)</h2>
<div class="card"><table><tr><th>label</th><th>share</th><th></th></tr>
{''.join(dist_rows)}</table></div>
{channel_html}
<h2>Per-stream evidence</h2>
<div class="card"><table>
<tr><th>stream</th><th>out bytes</th><th>blocks</th><th>label</th><th>conf</th>
<th>novelty&sigma;</th><th>signatures</th></tr>
{''.join(rows)}</table></div>
<h2>Investigator notes</h2>
<div class="card"><ul>{notes}</ul></div>
<div class="caveat"><b>Caveat.</b> This attribution is <b>supporting evidence</b>,
not proof of authorship. It identifies the compression implementation most
consistent with the bitstream. A program that internally links a common library
(e.g. zlib) will be attributed to that library, not to the program. Confidence
and an explicit &ldquo;{UNKNOWN}&rdquo; option are reported so the weight of the
evidence is transparent.</div>
</div></body></html>"""


def analysis_json(verdict: ArchiveVerdict, channel_compare: dict | None = None) -> str:
    d = verdict.to_dict()
    d["tool"] = {"name": "dfp", "version": __version__, "generated": _now()}
    if channel_compare:
        d["channel_comparison"] = channel_compare
    d["caveat"] = (
        "Supporting evidence, not proof of authorship. Attribution names the "
        "compression implementation most consistent with the bitstream."
    )
    return json.dumps(d, indent=2)


# --- evaluation report -----------------------------------------------------


def evaluation_html(result: EvalResult, aux: dict | None = None) -> str:
    conf_svg = confusion_svg(
        [[float(c) for c in row] for row in result.confusion],
        result.classes,
        "Closed-set confusion (row-normalised)",
    )
    pc_rows = "".join(
        f"<tr><td>{_esc(m.label)}</td><td>{m.support}</td><td>{m.precision:.2f}</td>"
        f"<td>{m.recall:.2f}</td><td>{m.f1:.2f}</td></tr>"
        for m in result.per_class
    )
    sizes = [c["size"] for c in result.size_curve]
    curve_svg = line_svg(
        sizes,
        {
            "acc on answered": [c["accuracy_on_answered"] for c in result.size_curve],
            "coverage": [c["coverage"] for c in result.size_curve],
            "raw accuracy": [c["raw_accuracy"] for c in result.size_curve],
        },
        "Minimum-evidence curve",
        xlabel="uncompressed bytes available",
        ylabel="rate",
        logx=True,
    )
    feat_svg = bar_svg(
        [n for n, _ in result.top_features[:10]],
        [v for _, v in result.top_features[:10]],
        "Top feature importances",
        ylabel="split share",
        fmt="{:.3f}",
    )
    os = result.open_set
    os_html = (
        f"unseen encoder <code>{_esc(os.get('unseen_encoder'))}</code>: "
        f"rejected {os.get('rejected')}/{os.get('n_streams')} "
        f"({os.get('rejection_rate', 0):.0%}), "
        f"misattributed {os.get('misattributed')}"
    )
    aux_html = ""
    if aux:
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{v['accuracy']:.2f}</td>"
            f"<td>{v['macro_f1']:.2f}</td><td class='small'>{_esc(v['classes'])}</td></tr>"
            for k, v in aux.items()
        )
        aux_html = f"""<h2>Auxiliary inference tasks (zlib)</h2>
        <div class="card"><table><tr><th>task</th><th>accuracy</th><th>macro-F1</th>
        <th>classes</th></tr>{rows}</table>
        <p class="small">These show the bitstream also reveals the zlib
        <i>strategy</i> and compression <i>level band</i>, not just the family.</p></div>"""

    notes = "".join(f"<li>{_esc(n)}</li>" for n in result.notes)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>DEFLATE provenance evaluation</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>DEFLATE Provenance &mdash; Evaluation</h1>
<div class="meta">generated {_now()} &middot; dfp v{__version__} &middot;
train={result.n_train} test={result.n_test} &middot; calibration T={result.temperature:.2f}</div>
<div class="card">
<b>Closed-set accuracy:</b> {result.accuracy:.1%} &nbsp;
<b>Macro-F1:</b> {result.macro_f1:.3f} &nbsp;
<b>Coverage:</b> {result.coverage:.0%} &nbsp;
<b>Accuracy on answered:</b> {result.accuracy_on_answered:.1%}</div>
<h2>Confusion matrix</h2><div class="card">{conf_svg}</div>
<h2>Per-class metrics</h2><div class="card"><table>
<tr><th>class</th><th>support</th><th>precision</th><th>recall</th><th>F1</th></tr>
{pc_rows}</table></div>
<h2>Minimum-evidence curve</h2><div class="card">{curve_svg}
<p class="small">Accuracy rises with the number of compressed bytes available;
the abstain option withholds a verdict when evidence is too thin.</p></div>
<h2>Open-set behaviour</h2><div class="card">{os_html}
<p class="small">An encoder never seen in training must be rejected, not forced
into a known class &mdash; the forensic-soundness property.</p></div>
{aux_html}
<h2>Feature importance</h2><div class="card">{feat_svg}</div>
<h2>Notes</h2><div class="card"><ul>{notes}</ul></div>
</div></body></html>"""


def evaluation_json(result: EvalResult, aux: dict | None = None) -> str:
    d = {
        "tool": {"name": "dfp", "version": __version__, "generated": _now()},
        "classes": result.classes,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "coverage": result.coverage,
        "accuracy_on_answered": result.accuracy_on_answered,
        "confusion": result.confusion,
        "per_class": [
            {"label": m.label, "support": m.support, "precision": m.precision,
             "recall": m.recall, "f1": m.f1}
            for m in result.per_class
        ],
        "size_curve": result.size_curve,
        "open_set": result.open_set,
        "temperature": result.temperature,
        "top_features": [{"feature": n, "importance": v} for n, v in result.top_features],
        "notes": result.notes,
    }
    if aux:
        d["auxiliary_tasks"] = aux
    return json.dumps(d, indent=2)
