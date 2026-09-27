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


def split_sources(corpus: Corpus, test_fraction: float = 0.3, seed: int = 1
                  ) -> tuple[set[str], set[str]]:
    """Split source ids (stratified by origin prefix) into train and test."""
    by_kind: dict[str, list[str]] = defaultdict(list)
    for sid in sorted({r["source_id"] for r in corpus.rows if r.get("origin") == "corpus"}):
        by_kind[sid.split(":")[0]].append(sid)
    rng = random.Random(seed)
    train, test = set(), set()
    for kind, ids in sorted(by_kind.items()):
        rng.shuffle(ids)
        cut = int(round(len(ids) * test_fraction))
        test.update(ids[:cut])
        train.update(ids[cut:])
    return train, test


def _true_label(row: dict, pred: str, classes: list[str]) -> str:
    """Canonical true label: the predicted one if it is in the set, otherwise
    the first real-world profile of the set (never the team's own synthetic
    encoder, which is not a class)."""
    if pred in row["profiles"]:
        return pred
    real = [p for p in row["profiles"] if p in classes]
    return (real or row["profiles"])[0]


def closed_set_metrics(rows: list[dict], preds, classes: list[str]) -> dict:
    y_raw = [p.raw_top for p in preds]
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
    answered = [(p, r) for p, r in zip(preds, rows) if not p.abstained]
    acc_ans = (sum(1 for p, r in answered if p.label in r["profiles"]) / len(answered)
               if answered else 0.0)
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "macro_f1": float(np.mean([m["f1"] for m in per if m["support"]])) if per else 0.0,
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "accuracy_on_answered": acc_ans,
        "labels": labels,
        "confusion": conf,
        "confusion_with_unknown": conf_u,
        "per_profile": per,
    }


def size_band_metrics(rows: list[dict], preds, target: float = 0.9) -> dict:
    bands = []
    for lo, hi in SIZE_BANDS:
        sel = [(r, p) for r, p in zip(rows, preds) if lo <= r["compressed_size"] < hi]
        if not sel:
            continue
        ans = [(r, p) for r, p in sel if not p.abstained]
        bands.append({
            "band": _band_name(lo, hi), "lo": lo, "hi": hi, "n": len(sel),
            "accuracy": sum(1 for r, p in sel if p.raw_top in r["profiles"]) / len(sel),
            "coverage": len(ans) / len(sel),
            "accuracy_on_answered": (sum(1 for r, p in ans if p.label in r["profiles"]) / len(ans)
                                     if ans else 0.0),
        })
    reliable = None
    for i, b in enumerate(bands):
        judged = [x for x in bands[i:] if x["n"] >= 20]
        if judged and all(x["accuracy_on_answered"] >= target for x in judged):
            reliable = b["lo"]
            break
    return {"bands": bands, "reliability_target": target,
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
    mask = training_mask(corpus, exclude_profiles={profile}, sources=train_src)
    clf = train_model(corpus, mask=mask, n_estimators=n_estimators, seed=seed)
    sel = corpus.mask(lambda r: r["source_id"] in test_src and r["profiles"] == [profile])
    if not sel.any():
        return profile, None
    preds = clf.predict(corpus.X[sel])
    sizes = [r["compressed_size"] for r, m in zip(corpus.rows, sel) if m]
    wrong = Counter(p.label for p in preds if not p.abstained)
    big = [p for p, s in zip(preds, sizes) if s >= 1024]
    return profile, {
        "n": len(preds),
        "rejection_rate": sum(p.abstained for p in preds) / len(preds),
        "rejection_rate_ge_1kib": (sum(p.abstained for p in big) / len(big)) if big else None,
        "misattributed_to": dict(wrong.most_common()),
    }


def leave_one_encoder_out(corpus: Corpus, train_src: set[str], test_src: set[str],
                          n_estimators: int = 60, seed: int = 1, workers: int = 4) -> dict:
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

    dp = [p.label for p in clf.predict(corpus.X[sample])]
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
        summary[name] = {
            "accuracy": tot["correct"] / n if n else 0.0,
            "misattribution_rate": tot["wrong"] / n if n else 0.0,
            "no_answer_rate": tot["no answer"] / n if n else 0.0,
            "per_profile": {k: {o: v[o] / sum(v.values()) for o in ("correct", "wrong", "no answer")}
                            for k, v in sorted(per.items())},
        }
    return {"n_streams": len(rows), "methods": summary, "notes": notes}


def edit_outcome(verdict, entry: str) -> dict:
    """Did the mixed-encoder check catch an edited entry?

    ``flagged``: the archive is reported as mixed; ``localised``: the edited
    entry is the one set apart (its profile differs from every other
    attributed entry); ``evidence``: the edited entry was large enough to be
    attributed at all.
    """
    mixed = any(f.check == "mixed-encoders" and f.kind == "inconsistent"
                for f in verdict.findings)
    ed = next((e for e in verdict.entries if e.name == entry), None)
    others = {e.prediction.label for e in verdict.entries
              if e.name != entry and e.status == "attributed"}
    localised = bool(mixed and ed is not None and ed.status == "attributed"
                     and others and ed.prediction.label not in others)
    return {"flagged": mixed, "localised": localised,
            "evidence": ed is not None and ed.status == "attributed"}


# --- archive experiments ------------------------------------------------------------

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


def build_writer_archives(out_dir: Path, sources: list, per_archive: int = 3,
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
    from .realfiles import _largest_deflate_entry, _word_claim, edit_with_python
    from .zipwriter import impersonate

    metas = {m["id"]: m for m in corpus.manifest["sources"]}
    sharing = corpus.manifest.get("profile_sharing", {})

    def profile_of(writer: str, library: str) -> str:
        return sharing.get(writer, {}).get("profile", library)

    def pick(ids, n):
        ids = sorted(i for i in ids if i in metas)
        random.Random(7).shuffle(ids)
        return [regenerate_source(metas[i]) for i in ids[: n * 3]]

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

    genuine, imp, claim, edits = [], [], [], []
    for a in test_arch:
        data = a["path"].read_bytes()
        true_prof = profile_of(a["writer"], a["library"])
        v = analyse_archive(str(a["path"]), clf, data=data, reencode=False)
        md_pred = mb.predict([extract_streams(a["path"], data=data)])[0]
        genuine.append({"writer": a["writer"], "profile": true_prof, "dp": v.profile,
                        "dp_correct": v.profile == true_prof, "metadata": md_pred,
                        "metadata_correct": md_pred == a["writer"], "flagged": v.inconsistent})

        # 1. impersonate another writer's container metadata
        others = [w for w in writers if w != a["writer"]
                  and profile_of(w, next(x["library"] for x in test_arch if x["writer"] == w)) != true_prof]
        others = others or [w for w in writers if w != a["writer"]]
        pick_i = int(hashlib.sha256(a["path"].name.encode()).hexdigest(), 16)
        target = others[pick_i % len(others)]
        forged = impersonate(data, template_by_writer[target])
        vf = analyse_archive(str(a["path"]), clf, data=forged, reencode=False)
        md_f = mb.predict([extract_streams(a["path"], data=forged)])[0]
        target_prof = profile_of(target, next(x["library"] for x in test_arch if x["writer"] == target))
        imp.append({"writer": a["writer"], "impersonated": target,
                    "metadata_followed_forgery": md_f == target,
                    "dp_kept_attribution": vf.profile == v.profile,
                    "dp_disagrees_with_forged_writer": vf.profile not in (target_prof, UNKNOWN)})

        # 2. rewrite the claimed producer to one the streams contradict
        claim_text = "Microsoft Office Word" if true_prof == "zlib" else "LibreOffice/24.2.7.2$Linux_X86_64"
        forged_claim = (_word_claim(data) if claim_text.startswith("Microsoft")
                        else impersonate(data, data, producer_claim=claim_text))
        vc = analyse_archive(str(a["path"]), clf, data=forged_claim, reencode=False)
        claim.append({"writer": a["writer"], "claim": claim_text,
                      "dp_kept_attribution": vc.profile == v.profile,
                      "flagged": any(f.kind == "inconsistent" and f.check in ("producer", "option-bits")
                                     for f in vc.findings)})

        # 3. mixed-encoder edit: one part edited by a Python script (zlib), or by
        #    Go when the archive itself is zlib
        entry = _largest_deflate_entry(data)
        editor = "zlib" if true_prof != "zlib" else "go"
        edited = edit_with_python(data, entry, editor)
        ve = analyse_archive(str(a["path"]), clf, data=edited, reencode=False)
        edits.append({"writer": a["writer"], "editor": editor, "entry": entry,
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
            "metadata_baseline_accuracy": rate(genuine, "metadata_correct"),
            "false_alarm_rate": rate(genuine, "flagged"),
        },
        "metadata_impersonation": {
            "metadata_baseline_followed_forgery": rate(imp, "metadata_followed_forgery"),
            "dp_kept_attribution": rate(imp, "dp_kept_attribution"),
            "dp_disagrees_with_forged_writer": rate(imp, "dp_disagrees_with_forged_writer"),
        },
        "producer_rewrite": {
            "flagged": rate(claim, "flagged"),
            "dp_kept_attribution": rate(claim, "dp_kept_attribution"),
        },
        "mixed_encoder_edit": {
            "flagged": rate(edits, "flagged"),
            "localised": rate(edits, "localised"),
            "edited_entry_had_enough_evidence": rate(edits, "evidence"),
            "flagged_when_enough_evidence": rate(edits, "flagged", lambda i: i["evidence"]),
        },
        "writer_errors": writer_errors[:20],
        "per_writer": per_writer_stats,
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

    progress("splitting by source file")
    train_src, test_src = split_sources(corpus, test_fraction, seed)
    test_rows_mask = corpus.mask(lambda r: r["source_id"] in test_src and not r["synthetic"]
                                 and r.get("origin") == "corpus")
    train_mask = training_mask(corpus, sources=train_src)
    overlap = {r["source_id"] for r, m in zip(corpus.rows, train_mask) if m} & test_src
    result["split"] = {
        "train_sources": len(train_src), "test_sources": len(test_src),
        "train_rows": int(train_mask.sum()), "test_rows": int(test_rows_mask.sum()),
        "test_rows_sharing_a_training_source": len(overlap),
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
    result["closed_set"] = closed_set_metrics(test_rows, preds, clf.classes)
    result["size_bands"] = size_band_metrics(test_rows, preds)
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
            corpus, train_src, test_src, seed=seed, workers=workers)
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
            "closed_set": closed_set_metrics(rows2, p2, clf.classes),
            "size_bands": size_band_metrics(rows2, p2),
        }
    if real_files:
        from .realfiles import evaluate_real_files

        progress("real application files (false alarms, edits, producer rewrite)")
        result["real_files"] = evaluate_real_files(clf, real_files)
    if python_docs:
        from .aggregate import analyse_archive

        progress("python-docx documents (claim Word, written by zlib)")
        flagged = []
        for p in python_docs:
            v = analyse_archive(str(p), clf, reencode=True, reencode_limit=3)
            flagged.append(v.inconsistent)
        result["python_docx"] = {"files": len(python_docs),
                                 "flagged_rate": sum(flagged) / len(flagged) if flagged else None}
    result["zlib_version"] = zlib.ZLIB_RUNTIME_VERSION
    return result, clf
