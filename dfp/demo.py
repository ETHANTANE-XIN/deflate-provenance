"""End-to-end demonstration of the proposal's scenarios.

Run from the project root::

    python -m dfp.demo                   # uses the 'default' model (no training)
    python -m dfp.demo --model NAME      # any model name or file
    python -m dfp.demo --retrain         # build a small corpus and model instead

Case studies (written to ``reports/demo`` by default):

1. **Claimed origin does not match the compressed data** (proposal section I):
   a DOCX whose ``docProps/app.xml`` names Microsoft Word, written by
   Python's ``zipfile`` (zlib).  A real tool, python-docx, produces exactly
   this, and is used too when it is installed.
2. **Producer rewrite**: a genuine document (LibreOffice when installed) has
   its claimed producer changed to Microsoft Word without touching any
   compressed stream.
3. **Mixed-encoder edit**: an archive written by a non-zlib writer (Go's
   ``archive/zip`` or 7-Zip, whichever is installed) has one part edited and
   recompressed by a Python script (zlib); with only zlib writers available,
   a Python-written archive is edited by another installed encoder instead.
4. **Metadata robustness**: every forgeable ZIP field is rewritten; the
   bitstream feature vector does not move.
5. *Extension, not part of the proposal*: the padding covert channel and its
   detector.
"""

from __future__ import annotations

import argparse
import io
import random
import zipfile
import zlib
from pathlib import Path


def _print_verdict(title: str, v) -> None:
    print(f"\n== {title} ==")
    print(f"  verdict: {v.status}; profile: {v.profile}"
          + (f" (setting {v.setting})" if v.setting else "")
          + (f", closest known profile {v.closest}" if v.closest else "")
          + f", share {v.share:.0%}, confidence {v.confidence:.0%}")
    for f in v.findings:
        if f.kind != "info":
            print(f"  [{f.kind}] {f.check}: {f.text[:200]}")


def _write_report(v, out_dir: Path, stem: str) -> None:
    from .report import analysis_html, analysis_json

    (out_dir / f"{stem}.report.html").write_text(analysis_html(v), encoding="utf-8")
    (out_dir / f"{stem}.report.json").write_text(analysis_json(v), encoding="utf-8")


def _texts(n: int, seed: int = 99) -> list[bytes]:
    from .corpus import _text, _xml

    rng = random.Random(seed)
    return [(_xml if i % 2 else _text)(random.Random(rng.random()), 20000 + 5000 * i)
            for i in range(n)]


def _app_xml(app: str) -> bytes:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Properties xmlns='
            '"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            f'<Application>{app}</Application><AppVersion>16.0000</AppVersion></Properties>'
            ).encode()


def case_claimed_word(clf, out_dir: Path) -> None:
    from .aggregate import analyse_archive

    texts = _texts(3)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", b'<?xml version="1.0"?><Types/>')
        z.writestr("docProps/app.xml", _app_xml("Microsoft Office Word"))
        z.writestr("word/document.xml", texts[1])
        z.writestr("word/styles.xml", texts[2])
        z.writestr("word/settings.xml", texts[0])
    path = out_dir / "claims-word-written-by-python.docx"
    path.write_bytes(buf.getvalue())
    v = analyse_archive(str(path), clf)
    _print_verdict("1. DOCX claims Microsoft Word, written by Python's zipfile", v)
    _write_report(v, out_dir, path.stem)

    try:
        import docx
    except ImportError:
        print("  (install python-docx to repeat this with a real tool)")
        return
    d = docx.Document()
    for t in texts[0].decode().split("."):
        d.add_paragraph(t)
    p2 = out_dir / "python-docx.docx"
    d.save(p2)
    v2 = analyse_archive(str(p2), clf)
    _print_verdict("1b. A real python-docx document (it writes 'Microsoft Macintosh Word')", v2)
    _write_report(v2, out_dir, p2.stem)


def case_producer_rewrite(clf, out_dir: Path) -> None:
    from .aggregate import analyse_archive
    from .realfiles import _word_claim, make_libreoffice_samples, soffice_binary

    if soffice_binary():
        docs = make_libreoffice_samples(out_dir / "libreoffice", _texts(1, seed=5),
                                        formats=("docx",))
        genuine = docs[0]
        title = f"2a. Genuine LibreOffice document ({genuine.name})"
    else:
        # Without LibreOffice the demo writes a stand-in itself: zlib streams (as
        # LibreOffice writes) with a LibreOffice claim typed in by this demo.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("docProps/app.xml", _app_xml("LibreOffice/24.2.7.2$Linux_X86_64"))
            z.writestr("word/document.xml", _texts(1, seed=5)[0])
        genuine = out_dir / "libreoffice-like.docx"
        genuine.write_bytes(buf.getvalue())
        title = ("2a. LibreOffice-like stand-in (LibreOffice is not installed: zlib streams "
                 "with a LibreOffice claim written by this demo)")
    v = analyse_archive(str(genuine), clf)
    _print_verdict(title, v)
    forged = out_dir / "producer-rewritten-to-word.docx"
    forged.write_bytes(_word_claim(genuine.read_bytes()))
    vf = analyse_archive(str(forged), clf)
    _print_verdict("2b. Same document, claimed producer rewritten to Microsoft Word", vf)
    print(f"  attribution unchanged by the rewrite: {vf.profile == v.profile}")
    _write_report(vf, out_dir, forged.stem)


#: non-zlib ZIP writers tried for the mixed-encoder case, with their setting
_NON_ZLIB_WRITERS = (("go", "6"), ("7zip", "mx5"))


def case_mixed_edit(clf, out_dir: Path) -> None:
    from .aggregate import analyse_archive
    from .encoders import get_encoder
    from .evaluate import edit_outcome
    from .realfiles import edit_with_python, pick_editor

    texts = _texts(4, seed=11)
    entries = [("word/document.xml", texts[0]), ("word/styles.xml", texts[1]),
               ("word/settings.xml", texts[2]), ("word/numbering.xml", texts[3])]
    writer = next(((name, s) for name, s in _NON_ZLIB_WRITERS
                   if get_encoder(name).available()), None)
    if writer is not None:
        # a non-zlib writer's archive, one part edited by a Python script (zlib)
        name, setting = writer
        path = out_dir / f"written-by-{name}.docx"
        get_encoder(name).write_zip(entries, str(path), setting)
        editor, title = "zlib", f"3a. Archive written by {name}"
    else:
        # only zlib writers here: a Python-written archive, one part edited by
        # another installed encoder
        editor = pick_editor("zlib")
        if editor is None:
            print("\n== 3. mixed-encoder edit: skipped (no non-zlib encoder installed) ==")
            return
        path = out_dir / "written-by-python.docx"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for n, d in entries:
                z.writestr(n, d)
        title = "3a. Archive written by Python's zipfile"
    v = analyse_archive(str(path), clf)
    _print_verdict(title, v)
    edited = out_dir / f"{path.stem}-edited-by-{editor}.docx"
    edited.write_bytes(edit_with_python(path.read_bytes(), "word/settings.xml", editor))
    ve = analyse_archive(str(edited), clf)
    _print_verdict(f"3b. Same archive after word/settings.xml was edited and recompressed "
                   f"by {editor}", ve)
    print(f"  outcome: {edit_outcome(ve, 'word/settings.xml')}")
    _write_report(ve, out_dir, edited.stem)


def case_metadata_robustness(out_dir: Path) -> None:
    from .adversarial import metadata_robustness

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.xml", _texts(1, seed=3)[0])
        z.writestr("b.xml", _texts(1, seed=4)[0])
    path = out_dir / "robust.zip"
    path.write_bytes(buf.getvalue())
    r = metadata_robustness(str(path))
    print("\n== 4. Metadata robustness ==")
    print(f"  metadata changed by the rewrite: {r.metadata_changed}")
    print(f"  bitstream feature vector changed: {r.bitstream_changed} "
          f"(max delta {r.max_feature_delta:.1e})")


def case_covert_extension() -> None:
    from .covert import corpus_padding_score, embed_message_across_streams

    streams = []
    for i in range(24):
        data = (f"carrier stream {i} ".encode() * (30 + i))
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        streams.append(c.compress(data) + c.flush())
    clean = corpus_padding_score(streams)
    stego = embed_message_across_streams(streams, b"HIDDEN")
    ok = all(zlib.decompress(a, -15) == zlib.decompress(b, -15) for a, b in zip(streams, stego))
    dirty = corpus_padding_score(stego)
    print("\n== 5. Extension (beyond the proposal): padding covert channel ==")
    print(f"  decompressed output unchanged after embedding: {ok}")
    print(f"  clean padding score {clean['fraction_of_padded_nonzero']:.2f} "
          f"({clean['verdict']}); stego {dirty['fraction_of_padded_nonzero']:.2f} "
          f"({dirty['verdict']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dfp.demo")
    ap.add_argument("--model", default=None,
                    help="model name or file (default: the 'default' model)")
    ap.add_argument("--retrain", action="store_true",
                    help="build a small corpus and model instead of loading one")
    ap.add_argument("--out", default="reports/demo")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clf = None
    if not args.retrain:
        from .modelstore import load, resolve

        try:
            clf = load(args.model)
            print(f"model: {resolve(args.model)}")
        except FileNotFoundError as exc:
            if args.model:
                raise
            print(f"{exc}\nfalling back to training a small model...")
    if clf is None:
        from .corpus import build_corpus, synthetic_sources
        from .training import train_model

        print("building a small corpus and model...")
        corpus = build_corpus(synthetic_sources(per_combo=2))
        clf = train_model(corpus, n_estimators=80)
    print(f"profiles: {', '.join(clf.classes)}")

    case_claimed_word(clf, out_dir)
    case_producer_rewrite(clf, out_dir)
    case_mixed_edit(clf, out_dir)
    case_metadata_robustness(out_dir)
    case_covert_extension()
    print(f"\nreports written under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
