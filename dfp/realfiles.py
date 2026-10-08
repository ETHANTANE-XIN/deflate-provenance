"""Files saved by real applications (proposal sections III.B and III.C).

Applications whose encoder cannot be called as a library, such as Microsoft
Word, are profiled from documents the team saves with them:

* :func:`add_app_files` adds a directory of saved documents to a corpus as an
  application profile.  Files are split in half by a hash of their content:
  one half trains, the other is kept aside for testing.  Before creating a
  new profile it checks whether an existing library reproduces the files'
  streams byte for byte (LibreOffice is reproduced by zlib, for example); if
  so the files join that library's profile, exactly as Java joins zlib.
* :func:`evaluate_real_files` runs the three real-file tests: the
  false-alarm rate on genuine files, the mixed-encoder edit test (one part of
  each genuine document is edited by a Python script and recompressed), and
  the producer-rewrite test (the claimed producer is changed to a false
  value without touching any compressed stream).
* :func:`make_libreoffice_samples` produces genuine LibreOffice documents
  with a headless ``soffice`` so the tests can run on any machine that has
  LibreOffice; Word documents have to be saved by hand.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

from .aggregate import analyse_archive
from .baselines import zlib_reencode_matches
from .containers import METHOD_DEFLATE, extract_streams
from .corpus import Corpus
from .features import features_to_vector, stream_features
from .zipwriter import impersonate, read_zip, replace_entry

DOC_SUFFIXES = {".docx", ".xlsx", ".pptx", ".docm", ".odt", ".ods", ".odp", ".jar",
                ".apk", ".zip", ".epub"}


def list_documents(directory: str | Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*")
                  if p.is_file() and p.suffix.lower() in DOC_SUFFIXES)


def file_split(path: Path) -> str:
    """Deterministic half split by content hash: 'train' or 'holdout'."""
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "train" if digest[0] % 2 == 0 else "holdout"


# --- profiling application files --------------------------------------------


def _reproducing_library(raw: bytes, output: bytes) -> str | None:
    """Which panel library reproduces ``raw`` exactly (reference encoders)."""
    if zlib_reencode_matches(raw, output):
        return "zlib"
    from .encoders import list_encoders

    for enc in list_encoders(only_available=True, include_synthetic=False):
        if not enc.reference or enc.library == "zlib":
            continue
        try:
            if raw in enc.compress_many(output, enc.settings()).values():
                return enc.library
        except Exception:
            continue
    return None


def add_app_files(corpus: Corpus, directory: str | Path, name: str,
                  description: str = "") -> dict:
    """Add saved documents in ``directory`` to ``corpus`` as application ``name``."""
    directory = Path(directory)
    docs = list_documents(directory)
    if not docs:
        raise ValueError(f"no documents found in {directory}")
    streams = []
    for path in docs:
        split = file_split(path)
        report = extract_streams(path)
        for s in report.streams:
            record, feats = stream_features(s.payload, s.start_bit)
            if record.error or not record.n_tokens:
                continue
            streams.append((path, split, s, record, feats))

    # does an existing library reproduce these streams?  Only parts large
    # enough for encoders to diverge are checked: tiny parts can coincide
    # byte for byte across encoders (see aggregate.PROOF_MIN_BYTES)
    from .aggregate import PROOF_MIN_BYTES

    reproduced: dict[str | None, int] = {}
    large = [x for x in streams if x[3].compressed_bytes >= PROOF_MIN_BYTES] or streams
    for _, _, s, record, _ in large[:200]:
        lib = _reproducing_library(s.payload[: record.compressed_bytes], record.output)
        reproduced[lib] = reproduced.get(lib, 0) + 1
    checked = sum(reproduced.values())
    best = max((k for k in reproduced if k), key=lambda k: reproduced[k], default=None)
    shares = best is not None and reproduced[best] / checked >= 0.95
    profile = best if shares else name

    # documents saved from one template share byte-identical parts (theme,
    # font table, settings): a stream that also occurs in a training document
    # is training data, so the held-out half is never scored on memorised bytes
    train_hashes = {hashlib.sha256(s.payload).hexdigest()
                    for _, split, s, _, _ in streams if split == "train"}
    shared = 0
    X, rows = [], []
    for path, split, s, record, feats in streams:
        if split == "holdout" and hashlib.sha256(s.payload).hexdigest() in train_hashes:
            split = "train"
            shared += 1
        X.append(features_to_vector(feats))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        rows.append({
            "source_id": f"app:{name}:{digest}",
            "content_type": path.suffix.lower().lstrip("."),
            "out_size": record.out_size,
            "compressed_size": record.compressed_bytes,
            "profile": profile,
            "profiles": [profile],
            "settings": {profile: [f"{name}:saved"]},
            "labels": [f"{name}/saved"],
            "synthetic": False,
            "origin": "app",
            "split": split,
            "hash": hashlib.sha256(s.payload).hexdigest(),
            "file": str(path),
            "entry": s.name,
        })
    corpus.X = np.vstack([corpus.X, np.array(X)]) if len(corpus.X) else np.array(X)
    corpus.rows.extend(rows)
    info = {
        "profile": profile,
        "description": description,
        "files": len(docs),
        "streams": len(rows),
        "train_files": sum(1 for d in docs if file_split(d) == "train"),
        "holdout_files": sum(1 for d in docs if file_split(d) == "holdout"),
        # streams of held-out documents identical to a training document's,
        # moved to training
        "holdout_streams_shared_with_training": shared,
        # which documents were kept aside, so the team can test with exactly
        # those (paths relative to the directory, so subfolders are kept apart)
        "holdout_file_names": sorted(d.relative_to(directory).as_posix()
                                     for d in docs if file_split(d) == "holdout"),
        "reproduced_by": {str(k): v for k, v in reproduced.items()},
        "shares_library": shares,
    }
    corpus.manifest.setdefault("app_profiles", {})[name] = info
    return info


# --- LibreOffice samples ------------------------------------------------------


def soffice_binary() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def make_libreoffice_samples(out_dir: str | Path, texts: list[bytes],
                             formats=("docx", "odt", "xlsx")) -> list[Path]:
    """Convert plain texts (and CSV derived from them) with headless LibreOffice."""
    exe = soffice_binary()
    if exe is None:
        raise RuntimeError("LibreOffice (soffice) is not installed")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        profile = Path(td) / "profile"
        txts, csvs = [], []
        for i, t in enumerate(texts):
            text = t.decode("utf-8", "replace").replace("\x00", "")
            p = Path(td) / f"doc{i:03d}.txt"
            p.write_text(text, encoding="utf-8")
            txts.append(str(p))
            words = text.split()
            rows = [",".join(words[j : j + 6]).replace('"', "") for j in range(0, len(words), 6)]
            c = Path(td) / f"sheet{i:03d}.csv"
            c.write_text("\n".join(rows[:400]), encoding="utf-8")
            csvs.append(str(c))
        base = [exe, f"-env:UserInstallation=file://{profile}", "--headless", "--norestore"]
        for fmt in formats:
            inputs = csvs if fmt in ("xlsx", "ods") else txts
            subprocess.run(base + ["--convert-to", fmt, "--outdir", str(out), *inputs],
                           capture_output=True, timeout=1800)
    return list_documents(out)


# --- tests on real files -------------------------------------------------------


def _largest_deflate_entry(data: bytes) -> str | None:
    entries, _ = read_zip(data)
    deflated = [e for e in entries if e.method == METHOD_DEFLATE and e.name != "docProps/app.xml"]
    return max(deflated, key=lambda e: len(e.raw)).name if deflated else None


def edit_with_python(data: bytes, entry: str, editor: str = "zlib") -> bytes:
    """Append a sentence to one part and recompress it with ``editor``.

    ``editor="zlib"`` is the proposal's "Python script" (CPython zlib at its
    default level); any other adapter name uses that encoder instead.
    """
    from .deflate import parse_stream
    from .encoders import get_encoder

    entries, _ = read_zip(data)
    target = next(e for e in entries if e.name == entry)
    original = parse_stream(target.raw).output
    marker = b"<!-- edited -->" if original.lstrip().startswith(b"<") else b" edited"
    new = original + marker
    if editor == "zlib":
        comp = zlib.compressobj(6, zlib.DEFLATED, -15)
        raw = comp.compress(new) + comp.flush()
    else:
        enc = get_encoder(editor)
        raw = enc.compress(new, enc.settings()[len(enc.settings()) // 2]).raw_deflate
    return replace_entry(data, entry, new, raw)


def _claim_with_bits(data: bytes, producer: str, option_bits: int) -> bytes:
    """Rewrite the claimed producer and set the ZIP option bits every DEFLATE
    entry carries to ``option_bits``, leaving all DEFLATE streams intact (a
    careful forger copies the claimed writer's header habits too)."""
    entries, _ = read_zip(data)
    tmpl = next(e for e in entries if e.method == METHOD_DEFLATE)
    from dataclasses import replace as dc_replace

    from .zipwriter import write_zip

    fake_template = write_zip([dc_replace(tmpl, flags=(tmpl.flags & ~0x06) | (option_bits << 1))])
    return impersonate(data, fake_template, producer_claim=producer)


def _word_claim(data: bytes) -> bytes:
    """Rewrite the claimed producer to Microsoft Word and copy Word's option
    bits (super fast) onto every entry, leaving all DEFLATE streams intact."""
    return _claim_with_bits(data, "Microsoft Office Word", 3)


#: non-zlib encoders tried, in order, to edit a part of a zlib-written file
EDITOR_PREFERENCE = ("go", "7zip", "node", "libdeflate", "zlib-ng", "zopfli")


def pick_editor(avoid_profile: str) -> str | None:
    """An installed encoder whose profile differs from ``avoid_profile``.

    A Python script (zlib) edits files written by anything else; files that
    are themselves zlib are edited by the first available non-zlib encoder.
    """
    if avoid_profile != "zlib":
        return "zlib"
    from .encoders import get_encoder

    for name in EDITOR_PREFERENCE:
        try:
            enc = get_encoder(name)
        except KeyError:
            continue
        if enc.available() and enc.library != "zlib":
            return name
    return None


def rewrite_target(claim: str | None) -> tuple[str, int]:
    """A false producer claim the file's own claim contradicts.

    Word documents are rewritten to claim LibreOffice (with its 'normal'
    option bits); anything else is rewritten to claim Microsoft Word (with
    Word's 'super fast' bits), so the test never rewrites a claim to itself.
    """
    from .producers import match_producer

    prod = match_producer(claim)
    if prod is not None and prod.key == "microsoft-word":
        return "LibreOffice/24.2.7.2$Linux_X86_64", 0
    return "Microsoft Office Word", 3


def _editor_profile(editor: str) -> str:
    """The profile an editor's output belongs to."""
    if editor == "zlib":
        return "zlib"
    from .encoders import get_encoder

    return get_encoder(editor).library


def editor_reproduces(data: bytes, entry: str, editor: str) -> bool:
    """Does ``editor``'s encoder reproduce the *original* ``entry`` exactly?

    Then the file was written by that encoder, and an edit recompressed by it
    is byte-for-byte what the writer would have produced: no method can detect
    it.  Decided from the bytes, not from the model's verdict.
    """
    from .aggregate import reproducing_setting
    from .deflate import parse_stream

    target = next(e for e in read_zip(data)[0] if e.name == entry)
    output = parse_stream(target.raw).output
    if editor == "zlib":
        return bool(zlib_reencode_matches(target.raw, output))
    return reproducing_setting(_editor_profile(editor), target.raw, output) is not None


def evaluate_real_files(clf, files: list[Path], editor: str = "zlib",
                        cross_editor: str | None = "auto") -> dict:
    """False alarms, edit detection and producer-rewrite detection.

    Files are analysed exactly as ``dfp analyse`` does (exact re-encoding on).
    ``editor`` edits one part with a Python script (zlib); ``cross_editor``
    edits it with a non-zlib encoder ("auto" picks an installed one, None
    skips that test).  An edit is only scored where the editor's encoder does
    not reproduce the original part exactly (decided from the bytes, see
    ``editor_reproduces``): a zlib-written file recompressed by zlib is
    byte-for-byte what its writer would have produced, so no method can
    detect it (those files are counted as ``*_same_encoder_skipped``).
    """
    from .evaluate import edit_outcome

    if cross_editor == "auto":
        cross_editor = pick_editor("zlib")
    genuine = []
    edits = []
    cross = []
    rewrites = []
    skipped = {"edit": 0, "cross": 0}
    for path in files:
        data = path.read_bytes()
        v = analyse_archive(str(path), clf, data=data)
        genuine.append({
            "file": path.name, "claim": v.claims.get("producer"), "profile": v.profile,
            "setting": v.setting, "status": v.status, "flagged": v.inconsistent,
            "findings": [f.text for f in v.findings if f.kind == "inconsistent"],
        })
        entry = _largest_deflate_entry(data)
        if entry is None:
            continue
        for ed, bucket, key in ((editor, edits, "edit"), (cross_editor, cross, "cross")):
            if not ed:
                continue
            if editor_reproduces(data, entry, ed):
                skipped[key] += 1
                continue
            edited = edit_with_python(data, entry, ed)
            ve = analyse_archive(str(path), clf, data=edited)
            bucket.append({"file": path.name, "entry": entry, "editor": ed,
                           **edit_outcome(ve, entry),
                           "edited_entry_label": next(
                               (e.label for e in ve.entries if e.name == entry), None)})
        if any(e.name == "docProps/app.xml" for e in read_zip(data)[0]):
            target, bits = rewrite_target(v.claims.get("producer"))
            vr = analyse_archive(str(path), clf, data=_claim_with_bits(data, target, bits))
            rewrites.append({
                "file": path.name, "claim_before": v.claims.get("producer"),
                "claim_after": target, "profile_before": v.profile, "profile_after": vr.profile,
                "kept_attribution": vr.profile == v.profile,
                "flagged": any(f.kind == "inconsistent" and f.check in ("producer", "option-bits")
                               for f in vr.findings),
                "flagged_by_streams": any(f.kind == "inconsistent" and f.check == "producer"
                                          for f in vr.findings),
            })

    def rate(items, key):
        return sum(1 for i in items if i[key]) / len(items) if items else None

    return {
        "files": len(files),
        "false_alarm_rate": rate(genuine, "flagged"),
        "genuine": genuine,
        "edit_editor": editor,
        "edit_flagged_rate": rate(edits, "flagged"),
        "edit_localised_rate": rate(edits, "localised"),
        "edit_same_encoder_skipped": skipped["edit"],
        "edits": edits,
        "cross_editor": cross_editor,
        "cross_edit_flagged_rate": rate(cross, "flagged"),
        "cross_edit_localised_rate": rate(cross, "localised"),
        "cross_edit_same_encoder_skipped": skipped["cross"],
        "cross_edits": cross,
        "inconclusive_rate": (sum(1 for g in genuine if g["status"] == "inconclusive")
                              / len(genuine)) if genuine else None,
        "rewrite_flag_rate": rate(rewrites, "flagged"),
        "rewrite_flagged_by_streams_rate": rate(rewrites, "flagged_by_streams"),
        "rewrite_kept_attribution_rate": rate(rewrites, "kept_attribution"),
        "rewrites": rewrites,
    }
