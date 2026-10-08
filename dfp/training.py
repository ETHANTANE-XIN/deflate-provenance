"""Train a :class:`ProvenanceClassifier` from a :class:`~dfp.corpus.Corpus`.

Training rows are the corpus streams that have exactly one real-world profile
(streams whose bytes several profiles produce identically carry no evidence
for choosing between them, so they are left out of training and scored with
their whole label set during evaluation).  The team's own synthetic encoder
and the held-out half of any application files are never trained on.
"""

from __future__ import annotations

import numpy as np

from .corpus import Corpus
from .ml import ProvenanceClassifier


def training_mask(corpus: Corpus, exclude_profiles: set[str] | None = None,
                  sources: set[str] | None = None) -> np.ndarray:
    """Rows to train on.

    ``sources`` restricts the *corpus* rows to a set of source ids (the
    evaluation's training split).  Application rows added with ``dfp app``
    carry their own file split and are always included from their 'train'
    half, never from the 'holdout' half.
    """
    exclude_profiles = exclude_profiles or set()

    def keep(r: dict) -> bool:
        return (
            r["profile"] is not None
            and not r["synthetic"]
            and r.get("split") != "holdout"
            and r["profile"] not in exclude_profiles
            and (sources is None or r["source_id"] in sources or r.get("origin") == "app")
        )

    return corpus.mask(keep)


def profile_info(corpus: Corpus) -> dict:
    """Which programs, versions and settings make up each profile."""
    info: dict[str, dict] = {}
    sharing = corpus.manifest.get("profile_sharing", {})
    for enc in corpus.manifest.get("encoders", []):
        if enc.get("synthetic"):
            continue
        prog = enc["program"]
        profile = sharing.get(prog, {}).get("profile", enc["library"])
        entry = info.setdefault(profile, {"programs": {}, "reference_version": None})
        entry["programs"][prog] = {"version": enc["version"], "settings": enc["settings"]}
        if enc.get("reference") and enc["library"] == profile:
            entry["reference_version"] = enc["version"]
    for name, app in corpus.manifest.get("app_profiles", {}).items():
        profile = app.get("profile", name)
        entry = info.setdefault(profile, {"programs": {}, "reference_version": None})
        entry["programs"][name] = {"version": app.get("description", "application files"),
                                   "settings": ["saved documents"]}
    return info


def train_model(
    corpus: Corpus,
    mask: np.ndarray | None = None,
    n_estimators: int = 150,
    seed: int = 1,
) -> ProvenanceClassifier:
    mask = training_mask(corpus) if mask is None else mask
    rows = [r for r, m in zip(corpus.rows, mask) if m]
    X = corpus.X[mask]
    labels = [r["profile"] for r in rows]
    groups = [r["source_id"] for r in rows]
    settings = [r["settings"].get(r["profile"], ["?"]) for r in rows]
    sizes = [r["compressed_size"] for r in rows]
    clf = ProvenanceClassifier(n_estimators=n_estimators, seed=seed)
    clf.fit(X, labels, groups, settings=settings, compressed_sizes=sizes)
    clf.profile_info = profile_info(corpus)
    clf.training = training_info(corpus, rows, n_estimators, seed)
    return clf


def training_info(corpus: Corpus, rows: list[dict], n_estimators: int, seed: int) -> dict:
    """What a saved model was trained on, recorded inside the model file."""
    from datetime import datetime, timezone

    from . import __version__

    m = corpus.manifest
    return {
        "trained": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dfp_version": __version__,
        "trees": n_estimators,
        "seed": seed,
        "training_streams": len(rows),
        "training_sources": len({r["source_id"] for r in rows}),
        "corpus_created": m.get("created"),
        "corpus_sources": m.get("n_sources"),
        "corpus_platform": m.get("platform"),
        "app_profiles": sorted(m.get("app_profiles", {})),
    }
