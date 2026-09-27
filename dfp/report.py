"""Forensic report generation (HTML + JSON).

Two report kinds:

* the **analysis report** for one examined file (proposal III.A): the
  predicted profile with its version, setting and confidence, the main
  features behind it, every consistency finding (claimed producer, ZIP
  option bits, entries made by different encoders), the per-entry evidence
  including entries with insufficient evidence, and the stated limits of the
  conclusion;
* the **evaluation report** (proposal III.C): split check, per-profile
  metrics and confusion matrices, results by compressed-size band, setting
  inference, the unknown-encoder test, the three baselines, the archive
  experiments, the second test set and the real application files.

Everything is self-contained HTML (inline CSS and SVG).
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone

from . import __version__
from .aggregate import ATTRIBUTED, INSUFFICIENT, UNKNOWN_ENCODER, ArchiveVerdict
from .charts import bar_svg, confusion_svg, line_svg
from .containers import OPTION_BITS

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:0;color:#1a1a1a;background:#f5f6f8}
.wrap{max-width:1000px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;border-bottom:2px solid #3b7dd8;
padding-bottom:4px;margin-top:28px}
.meta{color:#666;font-size:12px}
.card{background:#fff;border:1px solid #e3e5e8;border-radius:8px;padding:16px;margin:12px 0;
overflow-x:auto}
.verdict{font-size:20px;font-weight:bold}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;color:#fff}
.ok{background:#2ca05a}.warn{background:#c0392b}.unknown{background:#777}.info{background:#3b7dd8}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #e3e5e8;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#eef2f8}
.bar{height:10px;background:#3b7dd8;border-radius:3px;display:inline-block}
.caveat{background:#fff8e1;border:1px solid #f0d78a;padding:10px;border-radius:6px;font-size:13px}
code{background:#f0f0f3;padding:1px 4px;border-radius:3px;font-size:12px}
.small{font-size:12px;color:#555}
li.inconsistent{color:#a93226}li.consistent{color:#1e7b45}li.cannot-confirm{color:#555}
"""

_KIND_TAG = {"inconsistent": "warn", "consistent": "ok", "cannot-confirm": "unknown",
             "info": "info"}


def _esc(s) -> str:
    return html.escape(str(s))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def _page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en-GB"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{_esc(title)}</title><style>{_CSS}</style></head><body>'
            f'<div class="wrap">{body}</div></body></html>')


# --- analysis report -------------------------------------------------------------

CAVEAT = (
    "This report does not prove tampering or authorship. It states whether the "
    "compressed data is consistent with known compressor profiles and with the "
    "origin the file claims, with a stated confidence, so that an examiner knows "
    "where to look further. A profile names the compression library, not the "
    "program that called it: many programs share one library. Because DEFLATE is "
    "lossless, a stream that was decompressed and recompressed carries no trace "
    "of its earlier encoder, so recompression itself is not detected; "
    "inconsistencies are reported instead."
)


def _verdict_tag(v: ArchiveVerdict) -> str:
    if v.inconsistent:
        return '<span class="tag warn">inconsistencies found</span>'
    if v.profile in ("unknown", INSUFFICIENT):
        return '<span class="tag unknown">inconclusive</span>'
    return '<span class="tag ok">consistent</span>'


def analysis_html(v: ArchiveVerdict) -> str:
    version = v.model.get("profile_versions", {}).get(v.profile) if v.model else None
    head = (
        f"<h1>DEFLATE Compression-Provenance Report</h1>"
        f"<div class='meta'>file: <code>{_esc(v.path)}</code> &middot; container: "
        f"{_esc(v.container)} &middot; generated {_now()} &middot; dfp v{__version__}</div>"
    )
    counts = v.to_dict(False)["counts"]
    summary = (
        f"<div class='card'><div class='verdict'>Profile: {_esc(v.profile)} {_verdict_tag(v)}</div>"
        f"<p>{'Reference version: <b>' + _esc(version) + '</b>. ' if version else ''}"
        f"{'Most likely setting: <b>' + _esc(v.setting) + '</b>. ' if v.setting else ''}"
        f"Share of the compressed-size vote: {v.share:.0%}; mean confidence of the "
        f"supporting entries: {v.confidence:.0%}.</p>"
        f"<p class='small'>{counts[ATTRIBUTED]} entries attributed, "
        f"{counts[UNKNOWN_ENCODER]} unknown encoder, {counts[INSUFFICIENT]} insufficient "
        f"evidence, {counts['error']} unreadable.</p></div>"
    )
    findings = "".join(
        f"<li class='{_esc(f.kind)}'><span class='tag {_KIND_TAG.get(f.kind, 'info')}'>"
        f"{_esc(f.kind)}</span> <b>{_esc(f.check)}</b>: {_esc(f.text)}"
        + (f"<br><span class='small'>entries: {_esc(', '.join(f.entries[:12]))}</span>"
           if f.entries else "") + "</li>"
        for f in v.findings
    ) or "<li>none</li>"
    claims = v.claims or {}
    bits = v.metadata_summary.get("option_bits_seen") or []
    channel = (
        "<table><tr><th></th><th>What the container claims (editable)</th>"
        "<th>What the compressed data shows (this tool)</th></tr>"
        f"<tr><td>producer</td><td>{_esc(claims.get('producer', 'none stated'))}"
        f"{' (' + _esc(claims['source']) + ')' if claims.get('source') else ''}</td>"
        f"<td>profile <b>{_esc(v.profile)}</b></td></tr>"
        f"<tr><td>ZIP option bits</td><td>{_esc(', '.join(OPTION_BITS[b] for b in bits) or '-')}</td>"
        f"<td>setting {_esc(v.setting or '-')}</td></tr>"
        f"<tr><td>other metadata</td><td class='small'>host {_esc(v.metadata_summary.get('host_systems'))}, "
        f"version made by {_esc(v.metadata_summary.get('version_made_by'))}, first timestamp "
        f"{_esc(v.metadata_summary.get('first_timestamp'))}</td><td class='small'>not used for "
        f"attribution</td></tr></table>"
    )
    feats = "".join(
        f"<tr><td><code>{_esc(t['feature'])}</code></td><td>{t['contribution']:+.3f}</td>"
        f"<td class='small'>{_esc(t['meaning'])}</td></tr>" for t in v.top_features
    )
    feats_html = (
        "<table><tr><th>feature</th><th>contribution to P(profile)</th><th>meaning</th></tr>"
        f"{feats}</table><p class='small'>Contributions come from tracing each "
        "attributed entry through the forest's trees, weighted by compressed size.</p>"
        if feats else "<p>No attributed entries.</p>"
    )
    dist = "".join(
        f"<tr><td>{_esc(k)}</td><td>{val:.1%}</td>"
        f"<td><span class='bar' style='width:{int(val*200)}px'></span></td></tr>"
        for k, val in sorted(v.vote_distribution.items(), key=lambda kv: -kv[1])
    )
    rows = []
    for e in v.entries:
        p = e.prediction
        top = "; ".join(x["feature"] for x in (p.explanation[:3] if p else []))
        rows.append(
            f"<tr><td>{_esc(e.name)}</td><td>{_esc(e.method)}</td><td>{e.compressed_size}</td>"
            f"<td>{_esc(e.status)}</td><td>{_esc(p.raw_top if p else '-')}</td>"
            f"<td>{_esc(p.setting or '-') if p else '-'}</td>"
            f"<td>{f'{p.confidence:.2f}' if p else '-'}</td>"
            f"<td>{f'{p.distance:.2f}' if p else '-'}</td>"
            f"<td class='small'>{_esc(top or '-')}</td>"
            f"<td class='small'>{_esc('yes' if e.zlib_matches else 'no' if e.zlib_matches is not None else '-')}</td>"
            f"<td class='small'>{_esc(e.status_reason)}</td></tr>")
    model = v.model or {}
    model_html = (
        f"<p class='small'>Profiles known to this model: {_esc(', '.join(model.get('profiles', [])))}. "
        f"An entry is reported as <i>unknown encoder</i> when its calibrated confidence is below "
        f"{model.get('min_confidence', '-')} or it lies farther than {model.get('max_distance', '-')} "
        f"from every known profile, and as <i>insufficient evidence</i> when it is stored or has "
        f"fewer than {model.get('min_evidence_bytes', '-')} compressed bytes. These thresholds "
        f"were set on held-out source files.</p>" if model else
        "<p class='small'>No model supplied: structure only.</p>")
    notes = "".join(f"<li>{_esc(n)}</li>" for n in v.notes)
    body = (
        head + summary
        + f"<h2>Consistency findings</h2><div class='card'><ul>{findings}</ul></div>"
        + f"<h2>Claimed origin versus compressed data</h2><div class='card'>{channel}</div>"
        + f"<h2>Main features behind the attribution</h2><div class='card'>{feats_html}</div>"
        + f"<h2>Vote weighted by compressed size</h2><div class='card'><table>"
          f"<tr><th>profile</th><th>share</th><th></th></tr>{dist}</table></div>"
        + "<h2>Per-entry evidence</h2><div class='card'><table><tr><th>entry</th><th>method</th>"
          "<th>compressed bytes</th><th>status</th><th>best profile</th><th>setting</th>"
          "<th>confidence</th><th>distance</th><th>top features</th><th>zlib re-encodes</th>"
          f"<th>note</th></tr>{''.join(rows)}</table></div>"
        + f"<h2>Model</h2><div class='card'>{model_html}</div>"
        + (f"<h2>Notes</h2><div class='card'><ul>{notes}</ul></div>" if notes else "")
        + f"<div class='caveat'><b>Limits of this conclusion.</b> {_esc(CAVEAT)}</div>"
    )
    return _page("Provenance Report", body)


def analysis_json(v: ArchiveVerdict) -> str:
    d = v.to_dict()
    d["tool"] = {"name": "dfp", "version": __version__, "generated": _now()}
    d["caveat"] = CAVEAT
    return json.dumps(d, indent=2, default=str)


# --- evaluation report -------------------------------------------------------------


def _metrics_table(cs: dict) -> str:
    rows = "".join(
        f"<tr><td>{_esc(m['profile'])}</td><td>{m['support']}</td><td>{m['precision']:.2f}</td>"
        f"<td>{m['recall']:.2f}</td><td>{m['f1']:.2f}</td></tr>" for m in cs["per_profile"])
    return ("<table><tr><th>profile</th><th>support</th><th>precision</th><th>recall</th>"
            f"<th>F1</th></tr>{rows}</table>")


def _bands(sb: dict, title: str) -> str:
    bands = sb["bands"]
    svg = line_svg(
        [max(b["lo"], 64) for b in bands],
        {"accuracy": [b["accuracy"] for b in bands],
         "coverage": [b["coverage"] for b in bands],
         "accuracy on answered": [b["accuracy_on_answered"] for b in bands]},
        title, xlabel="compressed size (lower edge of band, bytes)", ylabel="rate", logx=True)
    rows = "".join(
        f"<tr><td>{_esc(b['band'])}</td><td>{b['n']}</td><td>{b['accuracy']:.2f}</td>"
        f"<td>{b['coverage']:.2f}</td><td>{b['accuracy_on_answered']:.2f}</td></tr>" for b in bands)
    rel = sb.get("smallest_reliable_compressed_size")
    return (f"{svg}<table><tr><th>compressed size</th><th>n</th><th>accuracy</th><th>coverage</th>"
            f"<th>accuracy on answered</th></tr>{rows}</table><p class='small'>Smallest compressed "
            f"size from which every band reaches {sb['reliability_target']:.0%} accuracy on answered "
            f"streams: <b>{_esc(rel if rel is not None else 'not reached')}"
            f"{' bytes' if rel is not None else ''}</b>.</p>")


def evaluation_html(r: dict) -> str:
    cs = r["closed_set"]
    sp = r["split"]
    parts = [
        "<h1>DeflateProvenance: Evaluation</h1>",
        f"<div class='meta'>generated {_now()} &middot; dfp v{__version__} &middot; "
        f"zlib {r.get('zlib_version')}</div>",
        "<h2>Data and split</h2><div class='card'>"
        f"<p>{sp['train_sources']} training and {sp['test_sources']} test source files "
        f"({sp['train_rows']} and {sp['test_rows']} streams). Test streams sharing a source "
        f"file with training: <b>{sp['test_rows_sharing_a_training_source']}</b>. "
        f"{sp['ambiguous_test_rows']} test streams are produced identically by more than one "
        "profile and are scored against their whole label set.</p>"
        + _encoder_table(r) + "</div>",
        "<h2>Closed-set results (test sources)</h2><div class='card'>"
        f"<b>Accuracy</b> {cs['accuracy']:.1%} &nbsp; <b>Macro-F1</b> {cs['macro_f1']:.3f} &nbsp; "
        f"<b>Coverage</b> {cs['coverage']:.0%} &nbsp; <b>Accuracy on answered</b> "
        f"{cs['accuracy_on_answered']:.1%}" + _metrics_table(cs)
        + confusion_svg([[float(x) for x in row] for row in cs["confusion"]], cs["labels"],
                        "Confusion matrix (row-normalised)") + "</div>",
        "<h2>Accuracy against compressed size</h2><div class='card'>"
        + _bands(r["size_bands"], "Accuracy by compressed-size band") + "</div>",
    ]
    if r.get("settings"):
        rows = "".join(f"<tr><td>{_esc(k)}</td><td>{v['n']}</td><td>{v['accuracy']:.2f}</td></tr>"
                       for k, v in r["settings"].items())
        parts.append("<h2>Setting inference</h2><div class='card'><table><tr><th>profile</th>"
                     f"<th>n</th><th>setting accuracy (set-aware)</th></tr>{rows}</table></div>")
    loeo = r.get("leave_one_encoder_out") or {}
    if loeo:
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{v['n']}</td><td>{v['rejection_rate']:.0%}</td>"
            f"<td>{_pct(v.get('rejection_rate_ge_1kib'))}</td>"
            f"<td class='small'>{_esc(v['misattributed_to'])}</td></tr>"
            for k, v in loeo.items() if v)
        su = r.get("synthetic_unknown")
        extra = (f"<p class='small'>The team's own encoder ({_esc(su['encoder'])}): "
                 f"{su['rejection_rate']:.0%} of {su['n']} streams rejected.</p>" if su else "")
        parts.append(
            "<h2>Unknown encoders (each profile left out of training)</h2><div class='card'>"
            "<table><tr><th>left-out profile</th><th>test streams</th><th>rejected as unknown</th>"
            f"<th>rejected (&ge; 1 KiB)</th><th>otherwise attributed to</th></tr>{rows}</table>"
            f"{extra}</div>")
    bl = r.get("baselines")
    if bl:
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{v['accuracy']:.0%}</td><td>{v['misattribution_rate']:.0%}</td>"
            f"<td>{v['no_answer_rate']:.0%}</td></tr>" for k, v in bl["methods"].items())
        per = ""
        profs = sorted(next(iter(bl["methods"].values()))["per_profile"])
        head = "".join(f"<th>{_esc(m)}</th>" for m in bl["methods"])
        for p in profs:
            per += f"<tr><td>{_esc(p)}</td>" + "".join(
                f"<td>{bl['methods'][m]['per_profile'].get(p, {}).get('correct', 0):.0%}</td>"
                for m in bl["methods"]) + "</tr>"
        notes = "".join(f"<li>{_esc(n)}</li>" for n in bl.get("notes", []))
        parts.append(
            f"<h2>Baselines on the same {bl['n_streams']} test streams</h2><div class='card'>"
            "<table><tr><th>method</th><th>correct</th><th>wrong profile</th><th>no answer</th></tr>"
            f"{rows}</table><p class='small'>Correct by true profile:</p><table><tr><th>profile</th>"
            f"{head}</tr>{per}</table><ul>{notes}</ul></div>")
    ar = r.get("archives")
    if ar and "genuine" in ar:
        g, im, pr, ed = (ar["genuine"], ar["metadata_impersonation"], ar["producer_rewrite"],
                         ar["mixed_encoder_edit"])
        rows = "".join(
            f"<tr><td>{_esc(w)}</td><td>{s['archives']}</td><td>{_pct(s['dp_accuracy'])}</td>"
            f"<td>{_pct(s['metadata_accuracy'])}</td><td>{_pct(s['false_alarm_rate'])}</td>"
            f"<td>{_pct(s['metadata_followed_forgery'])}</td><td>{_pct(s['dp_kept_attribution'])}</td>"
            f"<td>{_pct(s['claim_rewrite_flagged'])}</td><td>{_pct(s['edit_flagged'])}</td>"
            f"<td>{_pct(s['edit_localised'])}</td></tr>" for w, s in ar["per_writer"].items())
        parts.append(
            "<h2>Archive experiments (real ZIP writers)</h2><div class='card'>"
            f"<p>{ar['test_archives']} test archives from {len(ar['writers'])} writers "
            f"({_esc(', '.join(ar['writers']))}); the metadata baseline was trained on "
            f"{ar['train_archives']} archives built from the training sources.</p><ul>"
            f"<li>Genuine archives: DeflateProvenance profile correct {_pct(g['dp_accuracy'])}; "
            f"metadata baseline writer correct {_pct(g['metadata_baseline_accuracy'])}; false alarms "
            f"{_pct(g['false_alarm_rate'])}.</li>"
            f"<li>Metadata rewritten to impersonate another writer: the metadata baseline followed "
            f"the forgery {_pct(im['metadata_baseline_followed_forgery'])}; DeflateProvenance kept its "
            f"attribution {_pct(im['dp_kept_attribution'])} and disagreed with the forged writer "
            f"{_pct(im['dp_disagrees_with_forged_writer'])}.</li>"
            f"<li>Claimed producer rewritten to a false value: flagged {_pct(pr['flagged'])}; "
            f"attribution kept {_pct(pr['dp_kept_attribution'])}.</li>"
            f"<li>One part edited and recompressed by a different library: archive flagged as mixed "
            f"{_pct(ed['flagged'])}, edited entry singled out {_pct(ed['localised'])}; when the "
            f"edited entry had enough evidence to be attributed "
            f"({_pct(ed['edited_entry_had_enough_evidence'])} of cases) it was flagged "
            f"{_pct(ed['flagged_when_enough_evidence'])}.</li></ul>"
            "<table><tr><th>writer</th><th>archives</th><th>DP correct</th><th>metadata correct</th>"
            "<th>false alarms</th><th>metadata fooled</th><th>DP kept</th><th>claim flagged</th>"
            f"<th>edit flagged</th><th>edit localised</th></tr>{rows}</table></div>")
    st = r.get("second_test_set")
    if st:
        c2 = st["closed_set"]
        parts.append(
            "<h2>Second test set (a different corpus)</h2><div class='card'>"
            f"<p>{_esc(st.get('description'))} ({st.get('sources')} source files, never used in "
            f"training). Accuracy {c2['accuracy']:.1%}, macro-F1 {c2['macro_f1']:.3f}, coverage "
            f"{c2['coverage']:.0%}, accuracy on answered {c2['accuracy_on_answered']:.1%}.</p>"
            + _metrics_table(c2) + _bands(st["size_bands"], "Second test set by size") + "</div>")
    rf = r.get("real_files")
    if rf:
        rows = "".join(
            f"<tr><td>{_esc(g['file'])}</td><td class='small'>{_esc(g['claim'])}</td>"
            f"<td>{_esc(g['profile'])}</td><td>{_esc(g['setting'])}</td>"
            f"<td>{'yes' if g['flagged'] else 'no'}</td></tr>" for g in rf["genuine"])
        parts.append(
            "<h2>Real application files</h2><div class='card'>"
            f"<p>{rf['files']} genuine files. False-alarm rate (genuine file flagged as "
            f"inconsistent): <b>{_pct(rf['false_alarm_rate'])}</b>. One part edited with a Python "
            f"script ({_esc(rf['edit_editor'])}): flagged {_pct(rf['edit_flagged_rate'])}, "
            f"localised {_pct(rf['edit_localised_rate'])}; edited with "
            f"{_esc(rf['cross_editor'])}: flagged {_pct(rf['cross_edit_flagged_rate'])}, localised "
            f"{_pct(rf['cross_edit_localised_rate'])}. Claimed producer rewritten to Microsoft "
            f"Word without touching any stream: flagged {_pct(rf['rewrite_flag_rate'])}, "
            f"attribution kept {_pct(rf['rewrite_kept_attribution_rate'])}.</p>"
            "<table><tr><th>file</th><th>claimed producer</th><th>profile</th><th>setting</th>"
            f"<th>flagged</th></tr>{rows}</table></div>")
    pd = r.get("python_docx")
    if pd:
        parts.append(
            "<h2>python-docx documents</h2><div class='card'><p>python-docx writes "
            "'Microsoft Macintosh Word' into docProps/app.xml while compressing with zlib. "
            f"{pd['files']} such documents; flagged as inconsistent: <b>{_pct(pd['flagged_rate'])}</b>."
            "</p></div>")
    top = r.get("model", {}).get("top_features", [])[:10]
    if top:
        parts.append("<h2>Feature importance (mean decrease in impurity)</h2><div class='card'>"
                     + bar_svg([t["feature"] for t in top], [t["importance"] for t in top],
                               "Top features", ylabel="importance", fmt="{:.3f}") + "</div>")
    cal = r.get("model", {}).get("calibration", {})
    parts.append(f"<h2>Calibration (held-out training sources)</h2><div class='card'><pre class='small'>"
                 f"{_esc(json.dumps(cal, indent=1))}</pre></div>")
    return _page("DeflateProvenance Evaluation", "".join(parts))


def _encoder_table(r: dict) -> str:
    sharing = r["corpus"].get("profile_sharing") or {}
    rows = ""
    for e in r["corpus"].get("encoders", []):
        sh = sharing.get(e["program"])
        prof = sh["profile"] if sh else e["library"]
        verified = (f"{sh['identical_rate']:.0%} identical to {sh['reference']}" if sh else
                    "reference")
        rows += (f"<tr><td>{_esc(e['program'])}</td><td>{_esc(prof)}</td>"
                 f"<td class='small'>{_esc(e['version'])}</td>"
                 f"<td class='small'>{_esc(', '.join(e['settings']))}</td>"
                 f"<td class='small'>{_esc(verified)}</td></tr>")
    return ("<table><tr><th>program</th><th>profile</th><th>version</th><th>settings</th>"
            f"<th>profile sharing check</th></tr>{rows}</table>")


def evaluation_json(r: dict) -> str:
    return json.dumps({"tool": {"name": "dfp", "version": __version__, "generated": _now()},
                       **r}, indent=2, default=str)
