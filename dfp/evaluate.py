"""Evaluation harness (proposal section III.C, "Testing and Evaluation").

Everything is measured on data whose true encoder is known:

1. **Split by source file.**  Training and test sets are split by source
   file, so no source appears in both; the result records the check.
2. **Closed-set metrics** per profile: accuracy, precision, recall, macro-F1
   and confusion matrices; a stream that several profiles produce identically
   counts as correct if any of them is predicted.  Also coverage (how often
   the model answers rather than saying "unknown") and accuracy on answered.
3. **File-size bands**: the same metrics per compressed-size band, plotted
   against compressed size, and the smallest size that supports a reliable
   attribution.
4. **Setting inference** (for example the zlib level) with set-aware scoring.
5. **Unknown encoders**: each profile in turn is left out of training entirely
   and its test streams are scored for rejection versus misattribution; the
   team's own synthetic encoder is scored the same way.
6. **Baselines on the same streams**: preflate-rs's parameter estimate,
   brute-force re-encoding with 81 zlib settings, and (at archive level)
   container metadata alone.
7. **Archive experiments** with archives written by real ZIP writers: the
   metadata-only baseline, a metadata rewrite that impersonates another
   writer, a producer-claim rewrite, the mixed-encoder edit test and the
   false-alarm rate on genuine archives.
8. **Second test set** from a different corpus, and **real application
   files** (LibreOffice, python-docx, openpyxl, and Word when supplied).
"""

from __future__ import annotations

import hashlib
import os
import random
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .corpus import Corpus, regenerate_source, regenerate_streams
from .ml import UNKNOWN
from .training import train_model, training_mask

SIZE_BANDS = [(0, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 65536),
              (65536, 10**12)]


def _band_name(lo: int, hi: int) -> str:
    def f(x):
        return f"{x // 1024} KiB" if x >= 1024 else f"{x} B"
    return f">= {f(lo)}" if hi >= 10**12 else f"{f(lo)} - {f(hi)}"


#: near-duplicate threshold (estimated Jaccard similarity of 8-byte shingles)
FAMILY_JACCARD = 0.5
#: files in one directory whose names share at least this long a prefix are
#: treated as one family (crox1h.pcf / crox1hb.pcf, map-ISO8859-3 / -8)
FAMILY_PREFIX = 6
_SKETCH_K = 128


def content_sketch(data: bytes, k: int = _SKETCH_K) -> np.ndarray:
    """Bottom-k MinHash sketch of the 8-byte shingles of ``data``."""
    a = np.frombuffer(data, dtype=np.uint8)
    if len(a) < 8:
        return np.unique(a.astype(np.uint64))
    w = np.lib.stride_tricks.sliding_window_view(a, 8).astype(np.uint64)
    v = np.zeros(len(w), dtype=np.uint64)
    for i in range(8):
        v |= w[:, i] << np.uint64(8 * i)
    with np.errstate(over="ignore"):
        h = v * np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    u = np.unique(h)
    return u[:k]


def sketch_jaccard(a: np.ndarray, b: np.ndarray, k: int = _SKETCH_K) -> float:
    """Jaccard similarity estimated from two bottom-k sketches."""
    if not len(a) or not len(b):
        return 0.0
    union = np.union1d(a, b)[:k]
    both = np.intersect1d(np.intersect1d(union, a, assume_unique=True), b, assume_unique=True)
    return len(both) / len(union)


def source_families(corpus: Corpus) -> tuple[dict[str, str], dict]:
    """Group the corpus's real source files into families.

    Two real files are one family when they come from the same directory and
    their names share a prefix of at least ``FAMILY_PREFIX`` characters, or
    when their content is a near-duplicate (estimated Jaccard similarity of
    8-byte shingles of at least ``FAMILY_JACCARD``).  Generated sources each
    form their own family: every one has its own seed, and how far generated
    content generalises is what the second test set (a different corpus)
    measures.  Returns ``{source_id: family_id}`` and a summary.
    """
    metas = {m["id"]: m for m in corpus.manifest.get("sources", [])}
    ids = sorted({r["source_id"] for r in corpus.rows if r.get("origin") == "corpus"})
    parent = {s: s for s in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    real = [s for s in ids if not s.startswith("syn:") and s in metas]
    paths = {s: Path(metas[s]["origin"]) for s in real}
    sketches: dict[str, np.ndarray] = {}
    unreadable = 0
    for s in real:
        try:
            sketches[s] = content_sketch(regenerate_source(metas[s]).data)
        except Exception:
            unreadable += 1
    by_dir: dict[str, list[str]] = defaultdict(list)
    for s in real:
        by_dir[str(paths[s].parent)].append(s)
    by_name = 0
    for members in by_dir.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pre = os.path.commonprefix([paths[a].stem, paths[b].stem])
                if len(pre) >= FAMILY_PREFIX:
                    union(a, b)
                    by_name += 1
    by_content = 0
    keys = [s for s in real if s in sketches]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if sketch_jaccard(sketches[a], sketches[b]) >= FAMILY_JACCARD:
                union(a, b)
                by_content += 1
    fam = {s: find(s) for s in ids}
    sizes = Counter(fam[s] for s in real)
    return fam, {
        "real_sources": len(real),
        "families": len(sizes),
        "largest_family": max(sizes.values()) if sizes else 0,
        "pairs_joined_by_name": by_name,
        "pairs_joined_by_content": by_content,
        "unreadable_sources": unreadable,
        "jaccard_threshold": FAMILY_JACCARD,
        "name_prefix": FAMILY_PREFIX,
    }


def split_sources_with_families(corpus: Corpus, test_fraction: float = 0.3, seed: int = 1
                                ) -> tuple[set[str], set[str], dict]:
    """Split source ids into train and test by *family*, stratified by kind
    (generated versus real), so near-duplicate files never straddle the split."""
    fam, info = source_families(corpus)
    families: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sid, f in fam.items():
        families[sid.split(":")[0]][f].append(sid)
    rng = random.Random(seed)
    train, test = set(), set()
    for kind in sorted(families):
        groups = sorted(families[kind].values(), key=lambda g: g[0])
        rng.shuffle(groups)
        want = int(round(sum(len(g) for g in groups) * test_fraction))
        taken = 0
        for g in groups:
            if taken < want:
                test.update(g)
                taken += len(g)
            else:
                train.update(g)
    return train, test, info


def split_sources(corpus: Corpus, test_fraction: float = 0.3, seed: int = 1
                  ) -> tuple[set[str], set[str]]:
    """Split source ids into train and test by family (see
    :func:`split_sources_with_families`)."""
    train, test, _ = split_sources_with_families(corpus, test_fraction, seed)
    return train, test


def split_overlap_checks(corpus: Corpus, train_src: set[str], test_src: set[str]) -> dict:
    """Independent checks that no test content is in training.

    Counts test sources whose content digest equals a training source's, and
    test/training pairs that are near-duplicates by content sketch.  Both are
    measured on the regenerated content, not taken from the split itself.
    """
    metas = {m["id"]: m for m in corpus.manifest.get("sources", [])}
    digests: dict[str, str] = {}
    sketches: dict[str, np.ndarray] = {}
    for sid in train_src | test_src:
        if sid not in metas:
            continue
        try:
            data = regenerate_source(metas[sid]).data
        except Exception:
            continue
        digests[sid] = hashlib.sha256(data).hexdigest()
        if not sid.startswith("syn:"):
            sketches[sid] = content_sketch(data)
    train_digests = {digests[s] for s in train_src if s in digests}
    same = sum(1 for s in test_src if digests.get(s) in train_digests)
    near = 0
    tr = [s for s in train_src if s in sketches]
    for s in test_src:
        if s in sketches and any(sketch_jaccard(sketches[s], sketches[t]) >= FAMILY_JACCARD
                                 for t in tr):
            near += 1
    # application rows (dfp app) follow a per-document split: count held-out
    # streams whose bytes also occur in training
    train_hashes = {r["hash"] for r in corpus.rows
                    if (r.get("origin") == "app" and r.get("split") == "train")
                    or (r.get("origin") == "corpus" and r["source_id"] in train_src)}
    app_test = [r for r in corpus.rows if r.get("origin") == "app" and r.get("split") == "holdout"]
    return {"test_sources_identical_to_a_training_source": same,
            "test_sources_near_duplicate_of_a_training_source": near,
            "checked_sources": len(digests),
            "app_test_streams": len(app_test),
            "app_test_streams_identical_to_a_training_stream":
                sum(1 for r in app_test if r["hash"] in train_hashes)}


def selection_masks(corpus: Corpus, train_src: set[str], test_src: set[str],
                    exclude_profiles: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Training and test row masks.

    Corpus rows follow the source split; application rows (``dfp app``, for
    example Word documents) follow their own file split: the 'train' half
    trains and the 'holdout' half is tested, so application profiles are
    evaluated like every other profile.
    """
    train = training_mask(corpus, exclude_profiles=exclude_profiles, sources=train_src)
    test = corpus.mask(lambda r: not r["synthetic"] and (
        (r.get("origin") == "corpus" and r["source_id"] in test_src)
        or (r.get("origin") == "app" and r.get("split") == "holdout")))
    return train, test


def _true_label(row: dict, pred: str, classes: list[str]) -> str:
    """Canonical true label: the predicted one if it is in the set, otherwise
    the first real-world profile of the set (never the team's own synthetic
    encoder, which is not a class)."""
    if pred in row["profiles"]:
        return pred
    real = [p for p in row["profiles"] if p in classes]
    return (real or row["profiles"])[0]


def closed_set_metrics(rows: list[dict], preds, classes: list[str],
                       min_evidence: int = 0) -> dict:
    """Forced-choice and end-to-end metrics.

    End to end scores what ``dfp analyse`` reports for each stream: no answer
    when the open-set rule fires ("unknown encoder") or when the stream is
    smaller than the model's minimum evidence size ("insufficient evidence").
    """
    y_raw = [p.raw_top for p in preds]
    # the tool answers only when the open-set rule passes and the stream is
    # large enough to vote
    gives = [(not p.abstained) and r["compressed_size"] >= min_evidence
             for r, p in zip(rows, preds)]
    y_true = [_true_label(r, p, classes) for r, p in zip(rows, y_raw)]
    correct = [p in r["profiles"] for r, p in zip(rows, y_raw)]
    labels = sorted(set(classes) | set(y_true))
    idx = {c: i for i, c in enumerate(labels)}
    conf = [[0] * len(labels) for _ in labels]
    conf_u = [[0] * (len(labels) + 1) for _ in labels]
    for t, p, pr in zip(y_true, y_raw, preds):
        conf[idx[t]][idx[p]] += 1
        conf_u[idx[t]][len(labels) if pr.abstained else idx[p]] += 1
    per = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_raw) if p == c and t == c)
        fp = sum(1 for t, p in zip(y_true, y_raw) if p == c and t != c)
        fn = sum(1 for t, p in zip(y_true, y_raw) if p != c and t == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per.append({"profile": c, "support": sum(1 for t in y_true if t == c),
                    "precision": prec, "recall": rec, "f1": f1})
    answered = [(p, r) for p, r, g in zip(preds, rows, gives) if g]
    acc_ans = (sum(1 for p, r in answered if p.label in r["profiles"]) / len(answered)
               if answered else 0.0)

    # end to end: what the tool outputs.  "Unknown encoder" and "insufficient
    # evidence" are misses for recall and never false positives.
    e2e_correct = [g and p.label in r["profiles"] for r, p, g in zip(rows, preds, gives)]
    per_e2e = []
    for c in labels:
        tp = sum(1 for t, pr, g in zip(y_true, preds, gives) if g and pr.label == c and t == c)
        fp = sum(1 for t, pr, g in zip(y_true, preds, gives) if g and pr.label == c and t != c)
        fn = sum(1 for t, pr, g in zip(y_true, preds, gives) if t == c and (not g or pr.label != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_e2e.append({"profile": c, "support": sum(1 for t in y_true if t == c),
                        "precision": prec, "recall": rec, "f1": f1})
    return {
        "n": len(rows),
        # forced choice: the forest's top profile, the unknown rule switched off
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "macro_f1": float(np.mean([m["f1"] for m in per if m["support"]])) if per else 0.0,
        "per_profile": per,
        # end to end: the tool's real output (unknown counts as a miss)
        "end_to_end_accuracy": float(np.mean(e2e_correct)) if e2e_correct else 0.0,
        "end_to_end_macro_f1": (float(np.mean([m["f1"] for m in per_e2e if m["support"]]))
                                if per_e2e else 0.0),
        "per_profile_end_to_end": per_e2e,
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "accuracy_on_answered": acc_ans,
        "labels": labels,
        "confusion": conf,
        "confusion_with_unknown": conf_u,
        "min_evidence_bytes": min_evidence,
        "metric_note": ("'accuracy', 'macro_f1' and 'per_profile' are forced-choice (the "
                        "unknown rule and the minimum evidence size switched off); the "
                        "end_to_end_* figures, coverage and accuracy_on_answered score what "
                        "the tool actually outputs, counting 'unknown encoder' and "
                        "'insufficient evidence' as misses"),
    }


def size_band_metrics(rows: list[dict], preds, target: float = 0.9,
                      min_evidence: int = 0) -> dict:
    """Accuracy against compressed size.

    ``accuracy_on_answered`` is the classifier's accuracy where the open-set
    rule lets it answer (the minimum evidence size is what this measures, so
    it is not applied here); ``end_to_end_accuracy`` and ``coverage`` are what
    the tool reports, with the minimum evidence size applied.
    """
    bands = []
    for lo, hi in SIZE_BANDS:
        sel = [(r, p) for r, p in zip(rows, preds) if lo <= r["compressed_size"] < hi]
        if not sel:
            continue
        ans = [(r, p) for r, p in sel if not p.abstained]
        tool = [(r, p) for r, p in ans if r["compressed_size"] >= min_evidence]
        bands.append({
            "band": _band_name(lo, hi), "lo": lo, "hi": hi, "n": len(sel),
            "accuracy": sum(1 for r, p in sel if p.raw_top in r["profiles"]) / len(sel),
            "end_to_end_accuracy": sum(1 for r, p in tool if p.label in r["profiles"]) / len(sel),
            "coverage": len(tool) / len(sel),
            "classifier_coverage": len(ans) / len(sel),
            "accuracy_on_answered": (sum(1 for r, p in ans if p.label in r["profiles"]) / len(ans)
                                     if ans else 0.0),
        })
    # same rule as the classifier's minimum-evidence threshold: the lowest band
    # that meets the target on its own, with all larger streams together
    # meeting it too
    reliable = None
    for b in bands:
        if b["n"] < 20:
            continue
        above = [(r, p) for r, p in zip(rows, preds)
                 if r["compressed_size"] >= b["lo"] and not p.abstained]
        pooled = (sum(1 for r, p in above if p.label in r["profiles"]) / len(above)
                  if above else 0.0)
        if b["accuracy_on_answered"] >= target and pooled >= target:
            reliable = b["lo"]
            break
    return {"bands": bands, "reliability_target": target, "min_evidence_bytes": min_evidence,
            "smallest_reliable_compressed_size": reliable}


def setting_metrics(rows: list[dict], preds) -> dict:
    per: dict[str, list[bool]] = defaultdict(list)
    for r, p in zip(rows, preds):
        if p.abstained or p.label not in r["profiles"] or p.setting is None:
            continue
        per[p.label].append(p.setting in r["settings"].get(p.label, []))
    return {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(per.items())}


# --- unknown encoders ------------------------------------------------------------


def _loeo_job(args):
    corpus, profile, train_src, test_src, n_estimators, seed = args
    mask, test = selection_masks(corpus, train_src, test_src, exclude_profiles={profile})
    clf = train_model(corpus, mask=mask, n_estimators=n_estimators, seed=seed)
    sel = test & corpus.mask(lambda r: r["profiles"] == [profile])
    if not sel.any():
        return profile, None
    preds = clf.predict(corpus.X[sel])
    sizes = [r["compressed_size"] for r, m in zip(corpus.rows, sel) if m]
    wrong = Counter(p.label for p in preds if not p.abstained)
    big = [p for p, s in zip(preds, sizes) if s >= 1024]
    by_conf = sum(1 for p in preds if p.abstained and "confidence" in p.reason)
    by_nov = sum(1 for p in preds if p.abstained and "novelty" in p.reason)
    return profile, {
        "n": len(preds),
        "rejection_rate": sum(p.abstained for p in preds) / len(preds),
        "rejection_rate_ge_1kib": (sum(p.abstained for p in big) / len(big)) if big else None,
        "rejected_by_confidence_rule": by_conf / len(preds),
        "rejected_by_novelty_rule": by_nov / len(preds),
        "misattributed_to": dict(wrong.most_common()),
        "trees": n_estimators,
    }


def leave_one_encoder_out(corpus: Corpus, train_src: set[str], test_src: set[str],
                          n_estimators: int = 150, seed: int = 1, workers: int = 4) -> dict:
    profiles = corpus.profiles()
    jobs = [(corpus, p, train_src, test_src, n_estimators, seed) for p in profiles]
    out = {}
    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for prof, res in pool.map(_loeo_job, jobs):
                out[prof] = res
    else:
        for job in jobs:
            prof, res = _loeo_job(job)
            out[prof] = res
    return out


# --- baselines on the same streams ------------------------------------------------


def _zlib_job(raw: bytes) -> str:
    from .baselines import zlib_baseline_label

    return zlib_baseline_label(raw)


def baseline_comparison(corpus: Corpus, clf, test_src: set[str], per_profile: int = 40,
                        seed: int = 1, workers: int = 4) -> dict:
    """DeflateProvenance vs preflate-rs vs brute-force zlib on one sample."""
    from .baselines import preflate_binary, preflate_estimate

    rng = random.Random(seed)
    by_prof: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(corpus.rows):
        if (r["source_id"] in test_src and not r["synthetic"] and r["profile"]
                and r.get("origin") == "corpus"):
            by_prof[r["profile"]].append(i)
    sample = []
    for prof, ids in sorted(by_prof.items()):
        rng.shuffle(ids)
        sample += ids[:per_profile]
    rows = [corpus.rows[i] for i in sample]
    raws_by_hash = regenerate_streams(corpus, {r["source_id"] for r in rows})
    raws = [raws_by_hash[r["hash"]] for r in rows]

    # what the tool reports per stream: below the minimum evidence size it
    # gives no answer ("insufficient evidence")
    dp = [p.label if r["compressed_size"] >= clf.min_evidence_bytes else UNKNOWN
          for p, r in zip(clf.predict(corpus.X[sample]), rows)]
    if workers > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            zl = list(pool.map(_zlib_job, raws, chunksize=4))
    else:
        zl = [_zlib_job(r) for r in raws]
    methods = {"DeflateProvenance": dp, "brute-force zlib (81 settings)": zl}
    notes = []
    if preflate_binary():
        pf = preflate_estimate(raws)
        methods["preflate-rs 0.7.6 estimate"] = [x["label"] for x in pf]
    else:
        notes.append("preflate-rs baseline skipped: tools/preflate-estimate is not built")

    summary = {}
    for name, labels in methods.items():
        per: dict[str, Counter] = defaultdict(Counter)
        for r, lab in zip(rows, labels):
            outcome = ("correct" if lab in r["profiles"] else
                       "no answer" if lab in (UNKNOWN, "unknown") else "wrong")
            per[r["profile"]][outcome] += 1
        tot = Counter()
        for c in per.values():
            tot.update(c)
        n = sum(tot.values())
        # the narrower question every method can answer: "is this zlib?"
        is_z = [("zlib" in r["profiles"]) for r in rows]
        says_z = [lab == "zlib" for lab in labels]
        tp = sum(1 for a, b in zip(is_z, says_z) if a and b)
        fp = sum(1 for a, b in zip(is_z, says_z) if b and not a)
        fn = sum(1 for a, b in zip(is_z, says_z) if a and not b)
        summary[name] = {
            "accuracy": tot["correct"] / n if n else 0.0,
            "misattribution_rate": tot["wrong"] / n if n else 0.0,
            "no_answer_rate": tot["no answer"] / n if n else 0.0,
            "zlib_detector": {
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
            },
            "per_profile": {k: {o: v[o] / sum(v.values()) for o in ("correct", "wrong", "no answer")}
                            for k, v in sorted(per.items())},
        }
    notes.append("Brute-force zlib re-encoding can only answer 'zlib' or nothing: it is a "
                 "proof-level zlib detector (see zlib_detector), not a profile classifier, so "
                 "its overall accuracy is bounded by the share of zlib streams in the sample.")
    return {"n_streams": len(rows), "methods": summary, "notes": notes}


def edit_outcome(verdict, entry: str) -> dict:
    """Did the mixed-encoder check catch an edited entry?

    ``flagged``: the archive is reported as mixed; ``localised``: the report
    names the edited entry *and no untouched entry* as the odd one out;
    ``evidence``: the edited entry was large enough to be attributed at all.
    """
    mixed = [f for f in verdict.findings
             if f.check == "mixed-encoders" and f.kind == "inconsistent"]
    named = set(mixed[0].entries) if mixed else set()
    ed = next((e for e in verdict.entries if e.name == entry), None)
    return {"flagged": bool(mixed), "localised": bool(mixed) and named == {entry},
            "named_untouched_entries": len(named - {entry}),
            "named": sorted(named),
            "edited_entry_label": ed.label if ed is not None else None,
            "evidence": ed is not None and ed.status == "attributed"}


# --- archive experiments ------------------------------------------------------------


def _finding_list(verdict) -> list[dict]:
    """The findings that are not plain information, in a JSON-friendly form."""
    return [{"check": f.check, "kind": f.kind, "entries": list(f.entries)}
            for f in verdict.findings if f.kind != "info"]

_WRITER_SETTING = {"java": "6", "go": "6", "dotnet": "optimal", "7zip": "mx5",
                   "libarchive": "6"}


def _app_xml(app: str) -> bytes:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/'
            f'extended-properties"><Application>{app}</Application></Properties>').encode()


def _python_zip(entries, path):
    import zipfile

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)


#: source files per test archive: one per DOCX part name below, so an edit
#: touches one part of five, as in a real document
PARTS_PER_ARCHIVE = 5


def build_writer_archives(out_dir: Path, sources: list, per_archive: int = PARTS_PER_ARCHIVE,
                          errors: list | None = None) -> list[dict]:
    """DOCX-shaped archives written by every available native ZIP writer."""
    from .encoders import list_encoders

    writers = {"python-zipfile": ("zlib", _python_zip)}
    for enc in list_encoders(only_available=True, include_synthetic=False):
        if getattr(enc, "zip_writer", False):
            setting = _WRITER_SETTING.get(enc.name, enc.settings()[0])
            writers[enc.name] = (enc.library, (lambda e, s: lambda ents, p: e.write_zip(ents, p, s))(enc, setting))
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = []
    groups = [sources[i : i + per_archive] for i in range(0, len(sources) - per_archive + 1, per_archive)]
    for writer, (library, write) in writers.items():
        for gi, group in enumerate(groups):
            entries = [("[Content_Types].xml", b'<?xml version="1.0"?><Types/>' * 20),
                       ("docProps/app.xml", _app_xml(f"dfp test writer ({writer})"))]
            names = ["word/document.xml", "word/styles.xml", "word/media/part.bin",
                     "word/settings.xml", "word/numbering.xml"]
            entries += [(names[k], s.data) for k, s in enumerate(group)]
            path = out_dir / f"{writer}-{gi:03d}.docx"
            try:
                write(entries, str(path))
            except Exception as exc:
                if errors is not None:
                    errors.append(f"{writer}: {exc}"[:300])
                continue
            archives.append({"path": path, "writer": writer, "library": library,
                             "sources": [s.id for s in group]})
    return archives


def archive_experiments(corpus: Corpus, clf, train_src: set[str], test_src: set[str],
                        work_dir: Path, per_writer: int = 12) -> dict:
    from .aggregate import analyse_archive
    from .baselines import MetadataBaseline
    from .containers import extract_streams
    from .realfiles import _largest_deflate_entry, _word_claim, edit_with_python, pick_editor
    from .zipwriter import impersonate

    metas = {m["id"]: m for m in corpus.manifest["sources"]}
    sharing = corpus.manifest.get("profile_sharing", {})

    def profile_of(writer: str, library: str) -> str:
        return sharing.get(writer, {}).get("profile", library)

    def pick(ids, n):
        ids = sorted(i for i in ids if i in metas)
        random.Random(7).shuffle(ids)
        return [regenerate_source(metas[i]) for i in ids[: n * PARTS_PER_ARCHIVE]]

    writer_errors: list[str] = []
    train_arch = build_writer_archives(work_dir / "train", pick(train_src, per_writer),
                                       errors=writer_errors)
    test_arch = build_writer_archives(work_dir / "test", pick(test_src, per_writer),
                                      errors=writer_errors)
    if not test_arch:
        return {"note": "no ZIP writers available"}

    mb = MetadataBaseline().fit([extract_streams(a["path"]) for a in train_arch],
                                [a["writer"] for a in train_arch])
    writers = sorted({a["writer"] for a in test_arch})
    template_by_writer = {a["writer"]: a["path"].read_bytes() for a in train_arch}

    library_of = {a["writer"]: a["library"] for a in test_arch + train_arch}
    genuine, imp, claim, edits = [], [], [], []
    imp_skipped = edits_skipped = 0
    for a in test_arch:
        data = a["path"].read_bytes()
        true_prof = profile_of(a["writer"], a["library"])
        # the analyser exactly as shipped (exact re-encoding on)
        v = analyse_archive(str(a["path"]), clf, data=data)
        # ablation: no reference encoders, only CPython zlib (a bare Python install)
        v_zlib_only = analyse_archive(str(a["path"]), clf, data=data, verify=False)
        md_pred = mb.predict([extract_streams(a["path"], data=data)])[0]
        md_prof = profile_of(md_pred, library_of.get(md_pred, md_pred))
        genuine.append({"writer": a["writer"], "profile": true_prof, "dp": v.profile,
                        "dp_correct": v.profile == true_prof, "metadata": md_pred,
                        "metadata_correct": md_pred == a["writer"],
                        # same granularity as DeflateProvenance: the profile
                        "metadata_profile_correct": md_prof == true_prof,
                        "flagged": v.inconsistent,
                        "flagged_zlib_only": v_zlib_only.inconsistent,
                        "archive": a["path"].name,
                        "entries": {e.name: e.label for e in v.entries},
                        "findings": _finding_list(v)})

        # 1. impersonate another writer's container metadata
        others = [w for w in writers if w != a["writer"]
                  and profile_of(w, next(x["library"] for x in test_arch if x["writer"] == w)) != true_prof]
        others = [w for w in (others or [w for w in writers if w != a["writer"]])
                  if w in template_by_writer]
        if not others:
            imp_skipped += 1  # only one writer (or no template): nothing to impersonate
            others = None
        if others:
            pick_i = int(hashlib.sha256(a["path"].name.encode()).hexdigest(), 16)
            target = others[pick_i % len(others)]
            forged = impersonate(data, template_by_writer[target])
            vf = analyse_archive(str(a["path"]), clf, data=forged)
            md_f = mb.predict([extract_streams(a["path"], data=forged)])[0]
            target_prof = profile_of(target, library_of[target])
            imp.append({"writer": a["writer"], "impersonated": target,
                        "metadata_followed_forgery": md_f == target,
                        "dp_kept_attribution": vf.profile == v.profile,
                        "dp_disagrees_with_forged_writer": vf.profile not in (target_prof, UNKNOWN)})

        # 2. rewrite the claimed producer to one the streams contradict
        claim_text = "Microsoft Office Word" if true_prof == "zlib" else "LibreOffice/24.2.7.2$Linux_X86_64"
        forged_claim = (_word_claim(data) if claim_text.startswith("Microsoft")
                        else impersonate(data, data, producer_claim=claim_text))
        vc = analyse_archive(str(a["path"]), clf, data=forged_claim)
        claim.append({"writer": a["writer"], "claim": claim_text,
                      "dp_kept_attribution": vc.profile == v.profile,
                      "flagged": any(f.kind == "inconsistent" and f.check in ("producer", "option-bits")
                                     for f in vc.findings),
                      "flagged_by_streams": any(f.kind == "inconsistent" and f.check == "producer"
                                                for f in vc.findings)})

        # 3. mixed-encoder edit: one part edited by a Python script (zlib), or,
        #    when the archive itself is zlib, by an available non-zlib encoder
        entry = _largest_deflate_entry(data)
        editor = pick_editor(true_prof)
        if editor is None or entry is None:
            edits_skipped += 1  # no installed editor of another profile
            continue
        edited = edit_with_python(data, entry, editor)
        ve = analyse_archive(str(a["path"]), clf, data=edited)
        ve_zlib_only = edit_outcome(analyse_archive(str(a["path"]), clf, data=edited,
                                                    verify=False), entry)
        edits.append({"writer": a["writer"], "editor": editor, "entry": entry,
                      "archive": a["path"].name,
                      "entries": {e.name: e.label for e in ve.entries},
                      "flagged_zlib_only": ve_zlib_only["flagged"],
                      "localised_zlib_only": ve_zlib_only["localised"],
                      "findings": _finding_list(ve),
                      **edit_outcome(ve, entry)})

    def rate(items, key, filt=lambda x: True):
        sel = [i for i in items if filt(i)]
        return sum(1 for i in sel if i[key]) / len(sel) if sel else None

    per_writer_stats = {}
    for w in writers:
        f = (lambda i, w=w: i["writer"] == w)
        per_writer_stats[w] = {
            "archives": sum(1 for g in genuine if g["writer"] == w),
            "dp_accuracy": rate(genuine, "dp_correct", f),
            "metadata_accuracy": rate(genuine, "metadata_correct", f),
            "metadata_profile_accuracy": rate(genuine, "metadata_profile_correct", f),
            "false_alarm_rate": rate(genuine, "flagged", f),
            "metadata_followed_forgery": rate(imp, "metadata_followed_forgery", f),
            "dp_kept_attribution": rate(imp, "dp_kept_attribution", f),
            "claim_rewrite_flagged": rate(claim, "flagged", f),
            "edit_flagged": rate(edits, "flagged", f),
            "edit_localised": rate(edits, "localised", f),
        }
    return {
        "train_archives": len(train_arch),
        "test_archives": len(test_arch),
        "writers": writers,
        "genuine": {
            "dp_accuracy": rate(genuine, "dp_correct"),
            # same task as DeflateProvenance: name the profile
            "metadata_baseline_profile_accuracy": rate(genuine, "metadata_profile_correct"),
            # harder task, which only metadata can attempt: name the exact writer
            "metadata_baseline_writer_accuracy": rate(genuine, "metadata_correct"),
            "false_alarm_rate": rate(genuine, "flagged"),
            "false_alarm_rate_zlib_only": rate(genuine, "flagged_zlib_only"),
            "note": ("the genuine archives claim 'dfp test writer (...)', which is not in "
                     "the producer table, so this false-alarm rate covers the mixed-encoder "
                     "and duplicate-name checks; producer checks are measured on real "
                     "application files"),
        },
        "metadata_impersonation": {
            "metadata_baseline_followed_forgery": rate(imp, "metadata_followed_forgery"),
            "dp_kept_attribution": rate(imp, "dp_kept_attribution"),
            "dp_disagrees_with_forged_writer": rate(imp, "dp_disagrees_with_forged_writer"),
        },
        "producer_rewrite": {
            "flagged": rate(claim, "flagged"),
            "flagged_by_streams": rate(claim, "flagged_by_streams"),
            "dp_kept_attribution": rate(claim, "dp_kept_attribution"),
        },
        "mixed_encoder_edit": {
            "editors": dict(Counter(e["editor"] for e in edits)),
            "flagged": rate(edits, "flagged"),
            "localised": rate(edits, "localised"),
            "edited_entry_had_enough_evidence": rate(edits, "evidence"),
            "flagged_when_enough_evidence": rate(edits, "flagged", lambda i: i["evidence"]),
            # the same edits analysed without reference encoders (only CPython
            # zlib's exact checks), as on a machine with nothing else installed
            "flagged_zlib_only": rate(edits, "flagged_zlib_only"),
            "localised_zlib_only": rate(edits, "localised_zlib_only"),
        },
        "writer_errors": writer_errors[:20],
        "impersonation_skipped": imp_skipped,
        "edits_skipped_no_editor": edits_skipped,
        "per_writer": per_writer_stats,
        # every miss, so a reader can see which check fired on which entries
        "false_alarms": [{k: g[k] for k in ("archive", "writer", "dp", "entries", "findings")}
                         for g in genuine if g["flagged"]],
        "edits_missed_or_not_localised": [
            {k: e[k] for k in ("archive", "writer", "editor", "entry", "flagged", "named",
                               "edited_entry_label", "entries", "findings")}
            for e in edits if not e["localised"]],
    }


# --- the whole evaluation -------------------------------------------------------------


def run_evaluation(
    corpus: Corpus,
    work_dir: str | Path,
    test_fraction: float = 0.3,
    n_estimators: int = 150,
    seed: int = 1,
    second_corpus: Corpus | None = None,
    real_files: list[Path] | None = None,
    python_docs: list[Path] | None = None,
    loeo: bool = True,
    baselines: bool = True,
    archives: bool = True,
    workers: int | None = None,
    progress=print,
    cross_editor: str | None = "auto",
) -> tuple[dict, object]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    workers = workers or min(4, os.cpu_count() or 1)
    result: dict = {"corpus": {k: corpus.manifest.get(k) for k in
                               ("created", "dfp_version", "platform", "n_sources", "params",
                                "profile_sharing")}}
    result["corpus"]["encoders"] = [
        {k: e[k] for k in ("program", "library", "version", "settings")}
        for e in corpus.manifest.get("encoders", [])]

    progress("splitting by source family")
    train_src, test_src, families = split_sources_with_families(corpus, test_fraction, seed)
    train_mask, test_rows_mask = selection_masks(corpus, train_src, test_src)
    app_rows = corpus.mask(lambda r: r.get("origin") == "app")
    result["split"] = {
        "train_sources": len(train_src), "test_sources": len(test_src),
        "train_rows": int(train_mask.sum()), "test_rows": int(test_rows_mask.sum()),
        "app_train_rows": int((train_mask & app_rows).sum()),
        "app_test_rows": int((test_rows_mask & app_rows).sum()),
        "families": families,
        "overlap_checks": split_overlap_checks(corpus, train_src, test_src),
        "ambiguous_test_rows": sum(1 for r, m in zip(corpus.rows, test_rows_mask)
                                   if m and len(r["profiles"]) > 1),
    }

    progress("training on the training sources")
    clf = train_model(corpus, mask=train_mask, n_estimators=n_estimators, seed=seed)
    result["model"] = {"profiles": clf.classes, "calibration": clf.calibration,
                       "top_features": [{"feature": n, "importance": v}
                                        for n, v in clf.feature_importance()[:15]]}

    progress("scoring the test set")
    test_rows = [r for r, m in zip(corpus.rows, test_rows_mask) if m]
    preds = clf.predict(corpus.X[test_rows_mask])
    result["closed_set"] = closed_set_metrics(test_rows, preds, clf.classes,
                                              min_evidence=clf.min_evidence_bytes)
    result["size_bands"] = size_band_metrics(test_rows, preds,
                                             min_evidence=clf.min_evidence_bytes)
    result["settings"] = setting_metrics(test_rows, preds)

    syn = corpus.mask(lambda r: r["source_id"] in test_src and r["synthetic"])
    if syn.any():
        sp = clf.predict(corpus.X[syn])
        result["synthetic_unknown"] = {
            "encoder": "purepy (team's own encoder, never trained on)",
            "n": len(sp), "rejection_rate": sum(p.abstained for p in sp) / len(sp),
            "misattributed_to": dict(Counter(p.label for p in sp if not p.abstained)),
        }
    if loeo:
        progress("leave-one-encoder-out (unknown-encoder test)")
        result["leave_one_encoder_out"] = leave_one_encoder_out(
            corpus, train_src, test_src, n_estimators=n_estimators, seed=seed, workers=workers)
    if baselines:
        progress("baselines on the same streams")
        result["baselines"] = baseline_comparison(corpus, clf, test_src, seed=seed,
                                                  workers=workers)
    if archives:
        progress("archive experiments (metadata baseline, rewrites, edits)")
        result["archives"] = archive_experiments(corpus, clf, train_src, test_src,
                                                 work / "archives")
    if second_corpus is not None and len(second_corpus):
        progress("second test set (different corpus)")
        m2 = second_corpus.mask(
            lambda r: not r["synthetic"] and any(p in clf.classes for p in r["profiles"]))
        rows2 = [r for r, m in zip(second_corpus.rows, m2) if m]
        p2 = clf.predict(second_corpus.X[m2])
        result["second_test_set"] = {
            "description": second_corpus.manifest.get("params", {}).get("description", ""),
            "sources": second_corpus.manifest.get("n_sources"),
            "closed_set": closed_set_metrics(rows2, p2, clf.classes,
                                             min_evidence=clf.min_evidence_bytes),
            "size_bands": size_band_metrics(rows2, p2, min_evidence=clf.min_evidence_bytes),
        }
    if real_files:
        from .realfiles import evaluate_real_files

        progress("real application files (false alarms, edits, producer rewrite)")
        result["real_files"] = evaluate_real_files(clf, real_files, cross_editor=cross_editor)
    if python_docs:
        from .aggregate import analyse_archive

        progress("python-docx documents (claim Word, written by zlib)")
        flagged = []
        for p in python_docs:
            v = analyse_archive(str(p), clf)
            flagged.append(v.inconsistent)
        result["python_docx"] = {"files": len(python_docs),
                                 "flagged_rate": sum(flagged) / len(flagged) if flagged else None}
    result["zlib_version"] = zlib.ZLIB_RUNTIME_VERSION
    return result, clf
