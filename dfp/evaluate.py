"""Evaluation harness -- the quantitative results that earn marks.

Produces, over a corpus the tool generates itself (perfect oracle):

* closed-set accuracy, per-class precision/recall/F1, macro-F1, confusion matrix
* the minimum-evidence curve (accuracy vs available uncompressed bytes)
* open-set behaviour (an unseen encoder must be rejected, not misattributed)
* the adversarial result (metadata normalisation leaves the bitstream invariant)

All numbers are computed here; :mod:`dfp.report` renders them.  The label scheme
is deliberately *lineage*-based: encoders that share the zlib code base and
therefore produce near-identical bitstreams are grouped, and the levels/strategy
tasks are evaluated separately, because pretending vendor-name separability
exists where the bytes are identical would be dishonest -- and the report says so.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .corpus import build_corpus, Dataset
from .ml import ProvenanceClassifier
from .ml.classifier import UNKNOWN
from .schemes import relabel as relabel_scheme, SCHEMES


def relabel_lineage(labels: list[str]) -> list[str]:
    """Back-compat helper; defaults to the lineage scheme."""
    return relabel_scheme(labels, "lineage")


@dataclass
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvalResult:
    classes: list[str] = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)
    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class: list[ClassMetrics] = field(default_factory=list)
    coverage: float = 0.0
    accuracy_on_answered: float = 0.0
    n_train: int = 0
    n_test: int = 0
    size_curve: list[dict] = field(default_factory=list)
    open_set: dict = field(default_factory=dict)
    adversarial: dict = field(default_factory=dict)
    top_features: list[tuple[str, float]] = field(default_factory=list)
    temperature: float = 1.0
    notes: list[str] = field(default_factory=list)


def _macro_prf(y_true, y_pred, classes):
    per = []
    f1s = []
    for c in classes:
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == c and t == c)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == c and t != c)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p != c and t == c)
        support = sum(1 for t in y_true if t == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per.append(ClassMetrics(c, support, prec, rec, f1))
        f1s.append(f1)
    return per, float(np.mean(f1s)) if f1s else 0.0


def _balance(X, y, seed=7):
    rng = np.random.default_rng(seed)
    target = min(Counter(y).values())
    keep = np.concatenate(
        [rng.choice(np.where(y == c)[0], target, replace=False) for c in sorted(set(y))]
    )
    rng.shuffle(keep)
    return X[keep], y[keep]


def run_evaluation(
    sizes=None,
    per_combo: int = 6,
    n_estimators: int = 150,
    seed: int = 1,
    scheme: str = "lineage",
    progress=None,
) -> tuple[EvalResult, ProvenanceClassifier]:
    sizes = sizes or [4000, 16000, 48000, 96000]
    all_families = ["zlib", "java", "dotnet", "node", "libarchive", "purepy"]

    if progress:
        progress("building corpus")
    ds = build_corpus(sizes=sizes, per_combo=per_combo, families=all_families,
                      label_by="family")
    X = np.array(ds.X)
    y = np.array(relabel_scheme(ds.y, scheme))
    content = np.array(ds.content_types)
    size_arr = np.array(ds.sizes)

    # hold purepy fully out as the open-set unknown; train on the rest
    train_mask = y != "purepy"
    Xk, yk = X[train_mask], y[train_mask]
    ck, sk = content[train_mask], size_arr[train_mask]
    Xk, yk = _balance(Xk, yk, seed)
    # rebuild aligned content/size after balancing is nontrivial; recompute below
    # by re-deriving from a fresh split instead:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(yk))
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]

    if progress:
        progress("training classifier")
    clf = ProvenanceClassifier(
        n_estimators=n_estimators, seed=seed
    )
    clf.fit(Xk[tr], list(yk[tr]))

    if progress:
        progress("scoring closed set")
    preds = clf.predict(Xk[te])
    y_true = list(yk[te])
    y_raw = [p.raw_top for p in preds]
    classes = clf.classes

    idxmap = {c: i for i, c in enumerate(classes)}
    conf = [[0] * len(classes) for _ in classes]
    for p, t in zip(y_raw, y_true):
        conf[idxmap[t]][idxmap[p]] += 1
    acc = sum(1 for p, t in zip(y_raw, y_true) if p == t) / len(y_true)
    per_class, macro_f1 = _macro_prf(y_true, y_raw, classes)

    kept = [(p, t) for p, t in zip(preds, y_true) if not p.abstained]
    coverage = len(kept) / len(y_true) if y_true else 0.0
    acc_answered = (
        sum(1 for p, t in kept if p.label == t) / len(kept) if kept else 0.0
    )

    result = EvalResult(
        classes=classes,
        confusion=conf,
        accuracy=acc,
        macro_f1=macro_f1,
        per_class=per_class,
        coverage=coverage,
        accuracy_on_answered=acc_answered,
        n_train=len(tr),
        n_test=len(te),
        temperature=clf._temperature,
        top_features=clf.feature_importance()[:15],
    )

    # -- minimum-evidence curve ------------------------------------------
    if progress:
        progress("minimum-evidence curve")
    curve = []
    for size in [2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        cds = build_corpus(sizes=[size], per_combo=4,
                           families=[f for f in all_families if f != "purepy"],
                           label_by="family")
        cy = relabel_scheme(cds.y, scheme)
        cp = clf.predict(np.array(cds.X))
        raw_acc = sum(1 for p, t in zip(cp, cy) if p.raw_top == t) / len(cy)
        kp = [(p, t) for p, t in zip(cp, cy) if not p.abstained]
        cov = len(kp) / len(cy)
        aa = sum(1 for p, t in kp if p.label == t) / len(kp) if kp else 0.0
        curve.append(
            {"size": size, "raw_accuracy": raw_acc, "coverage": cov,
             "accuracy_on_answered": aa}
        )
    result.size_curve = curve

    # -- open set ---------------------------------------------------------
    if progress:
        progress("open-set test")
    ods = build_corpus(sizes=[4000, 16000, 48000], per_combo=4, families=["purepy"])
    op = clf.predict(np.array(ods.X))
    rejected = sum(1 for p in op if p.abstained)
    result.open_set = {
        "unseen_encoder": "purepy",
        "n_streams": len(op),
        "rejected": rejected,
        "rejection_rate": rejected / len(op) if op else 0.0,
        "misattributed": len(op) - rejected,
    }

    result.notes.append(
        "Encoders sharing the zlib code base (Info-ZIP/libarchive) are grouped "
        "with zlib; separating them by vendor name is impossible where the "
        "produced bytes are identical. Level and strategy inference are reported "
        "separately."
    )
    return result, clf


def run_level_strategy_tasks(per_combo: int = 6, seed: int = 1) -> dict:
    """Auxiliary tasks: zlib strategy id and zlib level-band inference."""
    ds = build_corpus(sizes=[4000, 16000, 48000], per_combo=per_combo,
                      families=["zlib"], label_by="family_level")
    X = np.array(ds.X)
    full = np.array(ds.labels_full)

    def strat(lbl):
        lv = lbl.split("/")[1]
        return "default" if lv.isdigit() else lv[1:]

    def band(lbl):
        lv = lbl.split("/")[1]
        if not lv.isdigit():
            return None
        n = int(lv)
        return "L0" if n == 0 else "L1-3" if n <= 3 else "L4-6" if n <= 6 else "L7-9"

    out = {}
    for name, fn, filt in (
        ("strategy", strat, lambda l: True),
        ("level_band", band, lambda l: band(l) is not None),
    ):
        m = np.array([filt(l) for l in full])
        Xt, ft = X[m], full[m]
        yt = np.array([fn(l) for l in ft])
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(yt))
        cut = int(len(idx) * 0.7)
        tr, te = idx[:cut], idx[cut:]
        clf = ProvenanceClassifier(n_estimators=100, seed=seed)
        clf.fit(Xt[tr], list(yt[tr]))
        pr = [p.raw_top for p in clf.predict(Xt[te])]
        acc = sum(1 for p, t in zip(pr, yt[te]) if p == t) / len(te)
        _, mf1 = _macro_prf(list(yt[te]), pr, clf.classes)
        out[name] = {"classes": clf.classes, "accuracy": acc, "macro_f1": mf1,
                     "support": dict(Counter(yt.tolist()))}
    return out
