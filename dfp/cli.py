"""Command-line interface for the DEFLATE provenance toolkit.

Subcommands
-----------
  inspect FILE            parse and dump the bitstream structure of every stream
  train [-o MODEL]        build a corpus and train the classifier, save the model
  analyse FILE -m MODEL   attribute a file, write HTML + JSON report
  evaluate [-o DIR]       run the full evaluation, write report + figures
  recompress FILE         constructive recompression test on each stream
  covert-embed / covert-detect   covert-channel demo and detector
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__


def _default_scratch() -> Path:
    root = os.environ.get("KIROCREW_SCRATCH") or os.environ.get("TMPDIR") or "."
    return Path(root)


def cmd_inspect(args) -> int:
    from .containers import extract_streams
    from .deflate import parse_stream

    report = extract_streams(args.file)
    print(f"file: {args.file}")
    print(f"container: {report.container}  streams: {len(report.streams)}  "
          f"stored: {report.stored_entries}")
    if report.metadata:
        keys = ("host_systems", "version_made_by", "first_timestamp",
                "extra_field_kinds", "os", "flevel_meaning")
        md = {k: report.metadata[k] for k in keys if k in report.metadata}
        print(f"metadata: {md}")
    for s in report.streams[: args.limit]:
        rec = parse_stream(s.payload, start_bit=s.start_bit, strict=False)
        print(f"  {s.label()[:50]:50s} blocks={rec.n_blocks:3d} "
              f"{rec.block_type_counts()} out={rec.out_size} "
              f"lit={rec.n_literals} match={rec.n_matches} "
              f"pad={rec.final_pad_bits} err={rec.error}")
    return 0


def cmd_train(args) -> int:
    import numpy as np

    from .corpus import build_corpus
    from .evaluate import _balance
    from .schemes import relabel as relabel_scheme
    from .ml import ProvenanceClassifier

    # In the coarse scheme purepy is a *class* (the non-standard/custom-encoder
    # exemplar), so it must be in training. In fine/lineage it is held out as the
    # open-set unknown and must NOT be trained on.
    if args.scheme == "coarse":
        families = ["zlib", "java", "dotnet", "node", "libarchive", "purepy"]
    else:
        families = ["zlib", "java", "dotnet", "node", "libarchive"]
    print("building corpus (this compresses many files with every encoder)...")
    ds = build_corpus(
        sizes=[4000, 16000, 48000, 96000], per_combo=args.per_combo,
        families=families, label_by="family",
        progress=lambda d, t: print(f"  {d}/{t}", end="\r"),
    )
    print()
    X = np.array(ds.X)
    y = np.array(relabel_scheme(ds.y, args.scheme))
    Xb, yb = _balance(X, y)
    clf = ProvenanceClassifier(n_estimators=args.trees, seed=1)
    clf.fit(Xb, list(yb))
    clf.save(args.out)
    print(f"trained on {len(yb)} streams, classes={clf.classes}")
    print(f"model saved to {args.out}")
    return 0


def cmd_analyse(args) -> int:
    from .adversarial import compare_channels
    from .aggregate import analyse_archive
    from .ml import ProvenanceClassifier
    from .report import analysis_html, analysis_json

    clf = ProvenanceClassifier.load(args.model) if args.model else None
    verdict = analyse_archive(args.file, clf)
    channel = compare_channels(args.file, verdict.top_label)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.file).stem
    html_path = out_dir / f"{stem}.report.html"
    json_path = out_dir / f"{stem}.report.json"
    html_path.write_text(analysis_html(verdict, channel), encoding="utf-8")
    json_path.write_text(analysis_json(verdict, channel), encoding="utf-8")

    print(f"attribution: {verdict.top_label} "
          f"(share {verdict.top_share:.0%}, confidence {verdict.confidence:.0%})")
    print(f"streams: {verdict.n_streams} attributed={verdict.n_attributed} "
          f"abstained={verdict.n_abstained} consistent={verdict.consistent}")
    for n in verdict.notes:
        print(f"  note: {n}")
    print(f"report: {html_path}")
    print(f"json:   {json_path}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import run_evaluation, run_level_strategy_tasks
    from .report import evaluation_html, evaluation_json

    result, clf = run_evaluation(
        per_combo=args.per_combo, n_estimators=args.trees, scheme=args.scheme,
        progress=lambda m: print(f"  {m}...", flush=True),
    )
    aux = run_level_strategy_tasks(per_combo=args.per_combo)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.html").write_text(
        evaluation_html(result, aux), encoding="utf-8")
    (out_dir / "evaluation.json").write_text(
        evaluation_json(result, aux), encoding="utf-8")
    if args.model:
        clf.save(args.model)

    print(f"\naccuracy={result.accuracy:.3f} macro-F1={result.macro_f1:.3f} "
          f"coverage={result.coverage:.2f} acc-on-answered={result.accuracy_on_answered:.3f}")
    print(f"open-set rejection={result.open_set['rejection_rate']:.2%}")
    print(f"strategy acc={aux['strategy']['accuracy']:.2f}  "
          f"level-band acc={aux['level_band']['accuracy']:.2f}")
    print(f"report: {out_dir / 'evaluation.html'}")
    return 0


def cmd_recompress(args) -> int:
    from .adversarial import detect_recompression
    from .containers import extract_streams

    report = extract_streams(args.file)
    for s in report.streams[: args.limit]:
        m = detect_recompression(s.payload)
        if m.matched:
            print(f"  {s.label()[:50]:50s} EXACT -> {m.encoder}")
        else:
            print(f"  {s.label()[:50]:50s} no exact match "
                  f"(best prefix {m.best_prefix_ratio:.2f})")
    return 0


def cmd_covert_embed(args) -> int:
    import zlib

    from .covert import bytes_to_bits, embed_in_padding
    from .deflate import parse_stream

    data = Path(args.file).read_bytes()
    comp = zlib.compressobj(6, zlib.DEFLATED, -15)
    raw = comp.compress(data) + comp.flush()
    bits = bytes_to_bits(args.message.encode())
    stego = embed_in_padding(raw, bits)
    Path(args.out).write_bytes(stego)
    # prove the output is unchanged
    same = zlib.decompress(stego, -15) == data
    rec = parse_stream(stego, strict=False)
    print(f"embedded {min(len(bits), rec.final_pad_bits)} bits into padding")
    print(f"decompressed output unchanged: {same}")
    print(f"stego stream: {args.out}")
    return 0


def cmd_covert_detect(args) -> int:
    from .covert import detect_covert_padding
    from .containers import extract_streams

    if args.file.endswith((".zip", ".docx", ".apk", ".jar", ".epub")):
        report = extract_streams(args.file)
        streams = [(s.label(), s.payload) for s in report.streams]
    else:
        streams = [(args.file, Path(args.file).read_bytes())]
    flagged = 0
    for name, raw in streams:
        f = detect_covert_padding(raw, name)
        if f.suspicious:
            flagged += 1
            print(f"  SUSPICIOUS {name}: {f.reason}")
    print(f"{flagged}/{len(streams)} streams flagged for non-zero padding")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dfp", description=__doc__)
    p.add_argument("--version", action="version", version=f"dfp {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("inspect", help="dump bitstream structure")
    q.add_argument("file")
    q.add_argument("--limit", type=int, default=50)
    q.set_defaults(func=cmd_inspect)

    q = sub.add_parser("train", help="build corpus and train the model")
    q.add_argument("-o", "--out", default="dfp_model.json")
    q.add_argument("--trees", type=int, default=150)
    q.add_argument("--per-combo", type=int, default=6)
    q.add_argument("--scheme", choices=["fine", "lineage", "coarse"],
                   default="coarse",
                   help="attribution granularity (default coarse: confident + court-usable)")
    q.set_defaults(func=cmd_train)

    q = sub.add_parser("analyse", help="attribute a file")
    q.add_argument("file")
    q.add_argument("-m", "--model", default=None)
    q.add_argument("-o", "--outdir", default="reports")
    q.set_defaults(func=cmd_analyse)

    q = sub.add_parser("evaluate", help="run the full evaluation")
    q.add_argument("-o", "--outdir", default="reports")
    q.add_argument("--trees", type=int, default=150)
    q.add_argument("--per-combo", type=int, default=6)
    q.add_argument("--scheme", choices=["fine", "lineage", "coarse"],
                   default="lineage")
    q.add_argument("--model", default=None)
    q.set_defaults(func=cmd_evaluate)

    q = sub.add_parser("recompress", help="constructive recompression test")
    q.add_argument("file")
    q.add_argument("--limit", type=int, default=20)
    q.set_defaults(func=cmd_recompress)

    q = sub.add_parser("covert-embed", help="hide a message in DEFLATE padding")
    q.add_argument("file")
    q.add_argument("message")
    q.add_argument("-o", "--out", default="stego.deflate")
    q.set_defaults(func=cmd_covert_embed)

    q = sub.add_parser("covert-detect", help="flag non-zero padding")
    q.add_argument("file")
    q.set_defaults(func=cmd_covert_detect)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
