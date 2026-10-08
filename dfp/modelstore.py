"""Model store: train once, then use the model by name.

A trained model is a single JSON file (``.json`` or compressed ``.json.gz``).
Instead of passing a path every time, save it under a name and refer to it by
that name; ``default`` is used when no model is given.  Names are looked up
in this order:

1. an existing file path, used as given;
2. the **local store**: ``$DFP_MODELS`` if set, otherwise ``~/.dfp/models``
   (``dfp train`` saves here, so a model you train yourself, for example one
   that includes Word documents, takes precedence);
3. the **bundled models** shipped in ``dfp/bundled_models`` with the
   repository, so a fresh clone can analyse files without training at all.

A model records the feature set it was trained on; if the code's features
change, loading it fails with a message to retrain rather than giving wrong
answers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_NAME = "default"
BUNDLED_DIR = Path(__file__).resolve().parent / "bundled_models"
_SUFFIXES = (".json.gz", ".json")


def store_dir() -> Path:
    """The local model store (created on first use)."""
    root = os.environ.get("DFP_MODELS") or str(Path.home() / ".dfp" / "models")
    return Path(root)


def _candidates(name: str) -> list[tuple[str, Path]]:
    out = []
    for where, d in (("local", store_dir()), ("bundled", BUNDLED_DIR)):
        for suffix in _SUFFIXES:
            out.append((where, d / f"{name}{suffix}"))
    return out


def resolve(name_or_path: str | None = None) -> Path:
    """Path of the model called ``name_or_path`` (``default`` if omitted)."""
    name = name_or_path or DEFAULT_NAME
    p = Path(name)
    if p.is_file():
        return p
    for _, cand in _candidates(name):
        if cand.is_file():
            return cand
    known = ", ".join(sorted({m["name"] for m in list_models()})) or "none"
    raise FileNotFoundError(
        f"no model called '{name}' (looked in {store_dir()} and {BUNDLED_DIR}; "
        f"available: {known}). Train one with `python -m dfp train --save-as {name}` "
        "or pass a model file path.")


def load(name_or_path: str | None = None):
    """Load a model by name or path."""
    from .ml import ProvenanceClassifier

    return ProvenanceClassifier.load(str(resolve(name_or_path)))


def save(clf, name: str = DEFAULT_NAME) -> Path:
    """Save ``clf`` into the local store under ``name`` (compressed)."""
    d = store_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json.gz"
    clf.save(str(path))
    plain = d / f"{name}.json"
    if plain.exists():  # a stale uncompressed copy would shadow nothing but confuse
        plain.unlink()
    return path


def bundled_profiles(name: str = DEFAULT_NAME) -> list[str]:
    """Profiles of the bundled model ``name`` (empty if there is none).

    Read from the file without building the model, so it works even when the
    bundled model was trained on an older feature set.
    """
    for suffix in _SUFFIXES:
        path = BUNDLED_DIR / f"{name}{suffix}"
        if path.is_file():
            return list(_summary(path, "bundled").get("profiles", []))
    return []


def _summary(path: Path, where: str) -> dict:
    import gzip

    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception as exc:  # pragma: no cover - unreadable file
        return {"name": path.name, "where": where, "path": str(path), "error": str(exc)}
    name = path.name
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    cal = blob.get("calibration", {})
    return {
        "name": name,
        "where": where,
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "profiles": blob.get("classes", []),
        "training": blob.get("training", {}),
        "held_out_accuracy": cal.get("raw_accuracy"),
        "min_evidence_bytes": blob.get("min_evidence_bytes"),
    }


def list_models() -> list[dict]:
    """Every model in the local store and the bundled directory."""
    out, seen = [], set()
    for where, d in (("local", store_dir()), ("bundled", BUNDLED_DIR)):
        if not d.is_dir():
            continue
        for path in sorted(d.iterdir()):
            if path.name.endswith(_SUFFIXES) and path.is_file():
                s = _summary(path, where)
                s["active"] = s["name"] not in seen  # first match wins
                seen.add(s["name"])
                out.append(s)
    return out
