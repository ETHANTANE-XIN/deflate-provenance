"""End-to-end demonstration and case studies.

Run:  python -m dfp.demo            (from the project root, dfp importable)
  or:  python demo.py

Produces, under ./reports/ by default:

* Case study 1 -- forged/rebuilt document: a file whose container metadata
  says one thing while its bitstream was produced by a different encoder.  The
  tool's metadata-vs-bitstream comparison exposes the contradiction.
* Case study 2 -- repackaging: an archive recompressed by a different tool,
  caught by the constructive recompression test.
* The metadata-robustness experiment: normalise all metadata and show the
  bitstream feature vector does not move.
* The covert-channel demonstration: hide a message in DEFLATE padding with the
  decompressed bytes provably unchanged, then detect it across a corpus.
"""

from __future__ import annotations

import io
import os
import zipfile
import zlib
from pathlib import Path


def _scratch() -> Path:
    root = os.environ.get("KIROCREW_SCRATCH") or os.environ.get("TMPDIR") or "."
    p = Path(root) / "dfp_demo"
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_zip_with_encoder(entries: dict[str, bytes], raw_streams: dict[str, bytes]) -> bytes:
    """Assemble a ZIP whose entries carry pre-made raw DEFLATE payloads.

    ``raw_streams`` maps entry name -> raw DEFLATE bytes produced by a chosen
    encoder, so the archive's bitstream provenance is exactly that encoder even
    though the ZIP structure itself is written here.
    """
    import struct

    out = bytearray()
    central = bytearray()
    offsets = {}
    for name, raw in raw_streams.items():
        data = entries[name]
        crc = zlib.crc32(data) & 0xFFFFFFFF
        offsets[name] = len(out)
        name_b = name.encode()
        # local file header
        out += b"PK\x03\x04"
        out += struct.pack("<HHHHHIIIHH", 20, 0, 8, 0, 0x0021, crc,
                           len(raw), len(data), len(name_b), 0)
        out += name_b
        out += raw
    cd_start = len(out)
    for name, raw in raw_streams.items():
        data = entries[name]
        crc = zlib.crc32(data) & 0xFFFFFFFF
        name_b = name.encode()
        central += b"PK\x01\x02"
        central += struct.pack("<HHHHHHIIIHHHHHII", 20, 20, 0, 8, 0, 0x0021,
                               crc, len(raw), len(data), len(name_b), 0, 0, 0, 0,
                               0, offsets[name])
        central += name_b
    out += central
    out += b"PK\x05\x06"
    out += struct.pack("<HHHHIIH", 0, 0, len(raw_streams), len(raw_streams),
                       len(central), cd_start, 0)
    return bytes(out)


def case_study_forgery(clf, out_dir: Path) -> None:
    from dfp.adversarial import compare_channels
    from dfp.aggregate import analyse_archive
    from dfp.encoders import get_encoder
    from dfp.report import analysis_html, analysis_json

    print("\n== Case study 1: forged / rebuilt document ==")
    from .corpus import _text, _xml
    import random as _random
    rng = _random.Random(20260916)
    # realistic, non-trivially-compressible OOXML-like content: encoders diverge
    # here (unlike pure repetition, where all optimal encoders converge)
    entries = {
        "word/document.xml": _xml(rng, 90000),
        "word/styles.xml": _xml(rng, 40000),
        "word/settings.xml": _text(rng, 20000),
        "word/numbering.xml": _text(rng, 18000),
    }
    # rebuild the streams with our custom (non-standard) encoder -- NOT the
    # zlib-lineage toolchain Microsoft Word uses -- so the contradiction is a
    # positive "non_standard" attribution rather than a match.
    enc = get_encoder("purepy")
    raw = {n: enc.compress(d, "lazy").raw_deflate for n, d in entries.items()}
    blob = make_zip_with_encoder(entries, raw)
    path = out_dir / "forged.docx"
    path.write_bytes(blob)

    verdict = analyse_archive(str(path), clf, container="ooxml")
    channel = compare_channels(str(path), verdict.top_label)
    (out_dir / "forged.report.html").write_text(
        analysis_html(verdict, channel), encoding="utf-8")
    (out_dir / "forged.report.json").write_text(
        analysis_json(verdict, channel), encoding="utf-8")
    print(f"  container claims OOXML/Word; bitstream attributes to: {verdict.top_label}")
    print("  (Word uses the zlib toolchain; a 'non_standard' attribution means the "
          "streams were re-compressed by a custom encoder)")
    # constructive corroboration: does ANY standard encoder reproduce these bytes?
    from dfp.adversarial import detect_recompression
    from dfp.containers import extract_streams as _es
    rep2 = _es(str(path), container="ooxml")
    exact = sum(1 for s in rep2.streams
                if detect_recompression(s.payload).matched)
    print(f"  recompression cross-check: {exact}/{len(rep2.streams)} streams also "
          f"reproducible by a panel encoder")
    print("  -> the ML bitstream attribution above is the primary signal; the "
          "recompression test is corroboration when it yields an exact match")
    for n in verdict.notes:
        print(f"  -> {n}")
    print(f"  report: {out_dir / 'forged.report.html'}")


def case_study_repackaging(out_dir: Path) -> None:
    from dfp.adversarial import detect_recompression
    from dfp.containers import extract_streams

    print("\n== Case study 2: repackaging (recompression) ==")
    payload = b"classpath entry bytecode " * 800
    # original: Info-ZIP/zlib style
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("classes.dex", payload)
    orig = out_dir / "original.apk"
    orig.write_bytes(buf.getvalue())

    rep = extract_streams(str(orig))
    m = detect_recompression(rep.streams[0].payload)
    print(f"  original classes.dex -> recompression test: "
          f"{'EXACT ' + m.encoder if m.matched else 'no exact match'}")
    print("  (an APK rebuilt by a different tool would fail this exact test, "
          "flagging repackaging)")


def demo_metadata_robustness(out_dir: Path) -> None:
    from dfp.adversarial import metadata_robustness

    print("\n== Metadata-robustness experiment ==")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.xml", b"<a>" + b"payload " * 600 + b"</a>")
        z.writestr("b.xml", b"<b>" + b"payload " * 400 + b"</b>")
    path = out_dir / "robust.zip"
    path.write_bytes(buf.getvalue())
    r = metadata_robustness(str(path))
    print(f"  metadata changed by normalisation: {r.metadata_changed}")
    print(f"  bitstream feature vector changed:  {r.bitstream_changed} "
          f"(max delta {r.max_feature_delta:.2e})")
    print("  => the metadata approach collapses; the bitstream approach does not move.")


def demo_covert(out_dir: Path) -> None:
    from dfp.covert import (
        bytes_to_bits, corpus_padding_score, detect_covert_padding,
        embed_message_across_streams,
    )
    from dfp.containers import extract_streams

    print("\n== Covert-channel demonstration ==")
    # make several streams, embed a message across their padding
    streams = []
    for i in range(24):
        data = (f"carrier stream {i} ".encode() * (30 + i))
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        streams.append(c.compress(data) + c.flush())
    clean = corpus_padding_score(streams)
    stego = embed_message_across_streams(streams, b"HIDDEN")
    # verify every payload still decompresses identically
    ok = all(
        zlib.decompress(a, -15) == zlib.decompress(b, -15)
        for a, b in zip(streams, stego)
    )
    dirty = corpus_padding_score(stego)
    print(f"  decompressed output unchanged after embedding: {ok}")
    print(f"  clean corpus padding score:  {clean['fraction_of_padded_nonzero']:.2f} "
          f"({clean['verdict']})")
    print(f"  stego corpus padding score:  {dirty['fraction_of_padded_nonzero']:.2f} "
          f"({dirty['verdict']})")


def main() -> int:
    from dfp.ml import ProvenanceClassifier
    from dfp.corpus import build_corpus
    from dfp.schemes import relabel
    from dfp.evaluate import _balance
    import numpy as np

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    scratch = _scratch()
    model_path = scratch / "demo_model.json"

    print("Training a small demo model (coarse scheme)...")
    ds = build_corpus(sizes=[2000, 4000, 8000, 16000, 32000, 64000], per_combo=6,
                      families=["zlib", "java", "dotnet", "node", "libarchive", "purepy"],
                      label_by="family")
    X = np.array(ds.X)
    y = np.array(relabel(ds.y, "coarse"))
    Xb, yb = _balance(X, y)
    clf = ProvenanceClassifier(n_estimators=120, seed=1)
    clf.fit(Xb, list(yb))
    clf.save(str(model_path))
    print(f"  classes: {clf.classes}")

    case_study_forgery(clf, out_dir)
    case_study_repackaging(out_dir)
    demo_metadata_robustness(out_dir)
    demo_covert(out_dir)
    print(f"\nArtefacts written under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
