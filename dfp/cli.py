"""Command-line interface for DeflateProvenance.

Subcommands
-----------
  corpus  -o DIR          build the reference corpus (every source x every encoder)
  app     CORPUS DOCS     add documents saved by an application (e.g. Word)
  train                   train the classifier and save it in the model store
  models                  list the trained models (local store and bundled)
  analyse FILE            examine a file with the default model, write HTML + JSON
  evaluate -o DIR         run the proposal's full evaluation
  inspect FILE            dump the DEFLATE structure of every stream
  reencode FILE           exact re-encoding test against every panel encoder
  producers               print the producer table
  encoders                list the available encoders and their versions
  covert-embed / covert-detect   padding covert channel (an extension)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

DEFAULT_SOURCES_HELP = (
    "directory of real files to use as extra sources (repeatable); files are "
    "decompressed if gzipped and skipped if already compressed"
)


def _progress(done: int, total: int) -> None:
    print(f"  {done}/{total} sources", end="\r", flush=True)


def _build_corpus(args):
    from .corpus import build_corpus, directory_sources, synthetic_sources

    sources = synthetic_sources(per_combo=args.per_combo, sizes=args.sizes)
    for d in args.sources or []:
        sources += directory_sources(d, limit=args.source_limit, label=f"dir{len(sources)}")
    print(f"building corpus from {len(sources)} sources "
          "(each is compressed with every available encoder and setting)...")
    corpus = build_corpus(sources, encoders=args.encoders, workers=args.workers,
                          progress=_progress,
                          params={"per_combo": args.per_combo, "sizes": args.sizes,
                                  "source_dirs": args.sources or []})
    print()
    return corpus


def cmd_corpus(args) -> int:
    corpus = _build_corpus(args)
    corpus.save(args.out)
    print(f"{len(corpus)} streams, profiles: {', '.join(corpus.profiles())}")
    for prog, info in corpus.manifest["profile_sharing"].items():
        print(f"  {prog}: {info['identical_rate']:.0%} byte-identical to "
              f"{info['reference']} -> profile '{info['profile']}'")
    if corpus.manifest["errors"]:
        print(f"  {len(corpus.manifest['errors'])} errors (see manifest.json)")
    print(f"corpus saved to {args.out}")
    return 0


def cmd_app(args) -> int:
    from .corpus import Corpus
    from .realfiles import add_app_files

    corpus = Corpus.load(args.corpus)
    info = add_app_files(corpus, args.docs, args.name, args.description)
    corpus.save(args.corpus)
    print(json.dumps(info, indent=2))
    if info["shares_library"]:
        print(f"'{args.name}' streams are reproduced by {info['profile']}: they join that profile")
    else:
        print(f"'{args.name}' becomes its own profile ({info['train_files']} files train, "
              f"{info['holdout_files']} held out for testing)")
    return 0


def cmd_train(args) -> int:
    from .corpus import Corpus
    from .training import train_model

    corpus = Corpus.load(args.corpus) if args.corpus else _build_corpus(args)
    clf = train_model(corpus, n_estimators=args.trees)
    print(f"profiles: {', '.join(clf.classes)}")
    cal = clf.calibration
    print(f"held-out calibration: accuracy {cal.get('raw_accuracy', 0):.3f} on "
          f"{cal.get('held_out_sources', 0)} source files; unknown if confidence < "
          f"{clf.min_confidence} or distance > {clf.max_distance:.2f}; insufficient evidence "
          f"below {clf.min_evidence_bytes} compressed bytes")
    if args.out:
        clf.save(args.out)
        print(f"model saved to {args.out}")
    if args.save_as or not args.out:
        from .modelstore import save

        name = args.save_as or "default"
        path = save(clf, name)
        print(f"model saved as '{name}' in the model store: {path}")
        if name == "default":
            print("`python -m dfp analyse FILE` will now use it without -m")
    return 0


def cmd_models(args) -> int:
    from .modelstore import BUNDLED_DIR, list_models, store_dir

    models = list_models()
    print(f"local store: {store_dir()}\nbundled:     {BUNDLED_DIR}\n")
    if not models:
        print("no models yet: run `python -m dfp train`")
        return 0
    for m in models:
        if "error" in m:
            print(f"{m['name']} ({m['where']}): unreadable: {m['error']}")
            continue
        t = m["training"]
        flag = "" if m["active"] else "  (shadowed by the local model of the same name)"
        print(f"{m['name']} [{m['where']}]{flag}")
        print(f"  file: {m['path']} ({m['size_mb']} MB)")
        print(f"  profiles: {', '.join(m['profiles'])}")
        if t:
            print(f"  trained {t.get('trained')} with dfp {t.get('dfp_version')} on "
                  f"{t.get('training_sources')} source files ({t.get('training_streams')} streams)")
        if m.get("held_out_accuracy") is not None:
            print(f"  held-out accuracy {m['held_out_accuracy']:.3f}; insufficient evidence "
                  f"below {m['min_evidence_bytes']} compressed bytes")
    return 0


def cmd_analyse(args) -> int:
    from .aggregate import analyse_archive
    from .report import analysis_html, analysis_json

    if args.no_model:
        clf = None
    else:
        from .modelstore import load, resolve

        clf = load(args.model)
        print(f"model: {resolve(args.model)}")
    v = analyse_archive(args.file, clf, reencode=not args.no_reencode)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.file).name
    html_path = out_dir / f"{stem}.report.html"
    json_path = out_dir / f"{stem}.report.json"
    html_path.write_text(analysis_html(v), encoding="utf-8")
    json_path.write_text(analysis_json(v), encoding="utf-8")

    print(f"profile: {v.profile}" + (f" (setting {v.setting})" if v.setting else "")
          + f"  share {v.share:.0%}  confidence {v.confidence:.0%}")
    counts = v.to_dict(False)["counts"]
    print("entries: " + ", ".join(f"{k} {n}" for k, n in counts.items() if n))
    for f in v.findings:
        print(f"  [{f.kind}] {f.check}: {f.text}")
    print(f"report: {html_path}\njson:   {json_path}")
    return 1 if (args.fail_on_inconsistent and v.inconsistent) else 0


def cmd_evaluate(args) -> int:
    from .corpus import Corpus, build_corpus, directory_sources
    from .evaluate import run_evaluation
    from .realfiles import list_documents, make_libreoffice_samples
    from .report import evaluation_html, evaluation_json

    corpus = Corpus.load(args.corpus) if args.corpus else _build_corpus(args)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    second = None
    second_sources = []
    if args.second_sources:
        second_sources = directory_sources(args.second_sources, limit=args.second_limit,
                                           label="second")
        print(f"second test set: {len(second_sources)} files from a different corpus")
        second = build_corpus(second_sources, workers=args.workers, progress=_progress,
                              params={"description": "files from " + ", ".join(args.second_sources)})
        print()

    real_files = list_documents(args.real) if args.real else []
    if args.libreoffice:
        texts = [s.data for s in second_sources[: args.libreoffice]] or [
            s.data for s in directory_sources(args.sources or [], limit=args.libreoffice)]
        if texts:
            print(f"generating LibreOffice documents from {len(texts)} texts...")
            real_files += make_libreoffice_samples(out_dir / "libreoffice", texts)
    python_docs = []
    if args.python_docx:
        python_docs = _python_docx_samples(out_dir / "python-docx", second_sources,
                                           args.python_docx)

    result, clf = run_evaluation(
        corpus, out_dir / "work", n_estimators=args.trees, second_corpus=second,
        real_files=real_files or None, python_docs=python_docs or None,
        loeo=not args.quick, baselines=not args.quick, archives=not args.quick,
        workers=args.workers, progress=lambda m: print(f"  {m}...", flush=True))
    (out_dir / "evaluation.html").write_text(evaluation_html(result), encoding="utf-8")
    (out_dir / "evaluation.json").write_text(evaluation_json(result), encoding="utf-8")
    if args.model:
        clf.save(args.model)
    if args.save_as:
        from .modelstore import save

        print(f"evaluated model saved as '{args.save_as}': {save(clf, args.save_as)}")
    cs = result["closed_set"]
    print(f"\naccuracy {cs['accuracy']:.3f}  macro-F1 {cs['macro_f1']:.3f}  "
          f"coverage {cs['coverage']:.2f}  accuracy on answered {cs['accuracy_on_answered']:.3f}")
    print(f"report: {out_dir / 'evaluation.html'}")
    return 0


def _python_docx_samples(out_dir: Path, sources, n: int) -> list[Path]:
    try:
        import docx  # python-docx
    except ImportError:
        print("python-docx not installed: skipping the python-docx check")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, s in enumerate(sources[:n]):
        d = docx.Document()
        text = s.data.decode("utf-8", "replace").replace("\x00", "")
        for para in text.split("\n")[:400]:
            d.add_paragraph("".join(ch for ch in para if ch.isprintable()))
        p = out_dir / f"python-docx-{i:03d}.docx"
        d.save(p)
        paths.append(p)
    return paths


def cmd_inspect(args) -> int:
    from .containers import extract_streams
    from .deflate import parse_stream

    report = extract_streams(args.file)
    print(f"file: {args.file}")
    print(f"container: {report.container}  DEFLATE streams: {len(report.streams)}  "
          f"stored entries: {report.stored_entries}")
    if report.claims:
        print(f"claims: {report.claims}")
    keys = ("host_systems", "version_made_by", "option_bits_seen", "first_timestamp",
            "extra_field_kinds", "os", "flevel_meaning")
    md = {k: report.metadata[k] for k in keys if k in report.metadata}
    if md:
        print(f"metadata: {md}")
    for s in report.streams[: args.limit]:
        rec = parse_stream(s.payload, start_bit=s.start_bit, strict=False)
        print(f"  {s.label()[:50]:50s} blocks={rec.n_blocks:3d} "
              f"{rec.block_type_counts()} out={rec.out_size} "
              f"lit={rec.n_literals} match={rec.n_matches} "
              f"pad={rec.final_pad_bits} err={rec.error}")
    return 0


def cmd_reencode(args) -> int:
    from .adversarial import reencode_panel
    from .containers import extract_streams

    report = extract_streams(args.file)
    for s in report.streams[: args.limit]:
        r = reencode_panel(s.payload)
        if r.error:
            print(f"  {s.label()[:50]:50s} {r.error}")
        elif r.matched:
            print(f"  {s.label()[:50]:50s} reproduced by: {', '.join(r.matches[:6])}"
                  + (f" (+{len(r.matches) - 6} more)" if len(r.matches) > 6 else ""))
        else:
            print(f"  {s.label()[:50]:50s} not reproduced by any panel encoder/setting")
    return 0


def cmd_producers(args) -> int:
    from .containers import OPTION_BITS
    from .producers import PRODUCERS

    for p in PRODUCERS:
        bits = ", ".join(OPTION_BITS[b] for b in sorted(p.option_bits)) if p.option_bits else "-"
        print(f"{p.label}\n  claim pattern: {p.pattern}\n  expected profiles: "
              f"{sorted(p.expected) if p.expected else '-'}  excluded: "
              f"{sorted(p.excluded) or '-'}  option bits: {bits}\n  basis: {p.basis}\n")
    return 0


def cmd_encoders(args) -> int:
    from .encoders import list_encoders

    for e in list_encoders(only_available=False):
        ok = e.available()
        print(f"{e.name:11s} library={e.library:14s} "
              f"{'available' if ok else 'missing  '}  {e.version() if ok else ''}")
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
    same = zlib.decompress(stego, -15) == data
    rec = parse_stream(stego, strict=False)
    print(f"embedded {min(len(bits), rec.final_pad_bits)} bits into padding")
    print(f"decompressed output unchanged: {same}")
    print(f"stego stream: {args.out}")
    return 0


def cmd_covert_detect(args) -> int:
    from .containers import extract_streams
    from .covert import detect_covert_padding

    try:
        report = extract_streams(args.file)
        streams = [(s.label(), s.payload) for s in report.streams]
    except Exception:
        streams = [(args.file, Path(args.file).read_bytes())]
    flagged = 0
    for name, raw in streams:
        f = detect_covert_padding(raw, name)
        if f.suspicious:
            flagged += 1
            print(f"  SUSPICIOUS {name}: {f.reason}")
    print(f"{flagged}/{len(streams)} streams flagged for non-zero padding")
    return 0


def _corpus_args(q) -> None:
    q.add_argument("--per-combo", type=int, default=4,
                   help="synthetic sources per content type and size (default 4)")
    q.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="synthetic source sizes in bytes")
    q.add_argument("--sources", action="append", help=DEFAULT_SOURCES_HELP)
    q.add_argument("--source-limit", type=int, default=200)
    q.add_argument("--encoders", nargs="+", default=None, help="restrict to these encoders")
    q.add_argument("--workers", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dfp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"dfp {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("corpus", help="build the reference corpus")
    q.add_argument("-o", "--out", default="corpus")
    _corpus_args(q)
    q.set_defaults(func=cmd_corpus)

    q = sub.add_parser("app", help="add documents saved by an application to a corpus")
    q.add_argument("corpus")
    q.add_argument("docs", help="directory of saved documents")
    q.add_argument("--name", required=True, help="application profile name, e.g. word")
    q.add_argument("--description", default="")
    q.set_defaults(func=cmd_app)

    q = sub.add_parser("train", help="train the classifier")
    q.add_argument("--corpus", default=None, help="corpus directory (built if omitted)")
    q.add_argument("--save-as", default=None,
                   help="name in the model store (default: 'default' unless -o is given)")
    q.add_argument("-o", "--out", default=None, help="also (or only) save to this file")
    q.add_argument("--trees", type=int, default=150)
    _corpus_args(q)
    q.set_defaults(func=cmd_train)

    q = sub.add_parser("models", help="list trained models")
    q.set_defaults(func=cmd_models)

    q = sub.add_parser("analyse", help="examine a file")
    q.add_argument("file")
    q.add_argument("-m", "--model", default=None,
                   help="model name or file (default: the 'default' model; see `dfp models`)")
    q.add_argument("--no-model", action="store_true",
                   help="structure and claims only, no attribution")
    q.add_argument("-o", "--outdir", default="reports")
    q.add_argument("--no-reencode", action="store_true",
                   help="skip the zlib re-encoding corroboration")
    q.add_argument("--fail-on-inconsistent", action="store_true",
                   help="exit with status 1 when an inconsistency is found")
    q.set_defaults(func=cmd_analyse)

    q = sub.add_parser("evaluate", help="run the full evaluation")
    q.add_argument("--corpus", default=None, help="corpus directory (built if omitted)")
    q.add_argument("-o", "--outdir", default="reports")
    q.add_argument("--trees", type=int, default=150)
    q.add_argument("--second-sources", nargs="+", default=None,
                   help="directories of a *different* corpus for the second test set "
                        "(for example a Govdocs1 sample)")
    q.add_argument("--second-limit", type=int, default=120)
    q.add_argument("--real", default=None,
                   help="directory of genuine application files (Word, LibreOffice, ...)")
    q.add_argument("--libreoffice", type=int, default=0,
                   help="generate this many LibreOffice documents per format as real files")
    q.add_argument("--python-docx", type=int, default=0,
                   help="generate this many python-docx documents (they claim Word)")
    q.add_argument("--model", default=None, help="also save the evaluated model to this file")
    q.add_argument("--save-as", default=None, help="also save it in the model store")
    q.add_argument("--quick", action="store_true",
                   help="skip the unknown-encoder, baseline and archive experiments")
    _corpus_args(q)
    q.set_defaults(func=cmd_evaluate)

    q = sub.add_parser("inspect", help="dump bitstream structure")
    q.add_argument("file")
    q.add_argument("--limit", type=int, default=50)
    q.set_defaults(func=cmd_inspect)

    q = sub.add_parser("reencode", help="exact re-encoding test against the panel")
    q.add_argument("file")
    q.add_argument("--limit", type=int, default=20)
    q.set_defaults(func=cmd_reencode)

    q = sub.add_parser("producers", help="print the producer table")
    q.set_defaults(func=cmd_producers)

    q = sub.add_parser("encoders", help="list encoders and versions")
    q.set_defaults(func=cmd_encoders)

    q = sub.add_parser("covert-embed", help="(extension) hide a message in DEFLATE padding")
    q.add_argument("file")
    q.add_argument("message")
    q.add_argument("-o", "--out", default="stego.deflate")
    q.set_defaults(func=cmd_covert_embed)

    q = sub.add_parser("covert-detect", help="(extension) flag non-zero padding")
    q.add_argument("file")
    q.set_defaults(func=cmd_covert_detect)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
