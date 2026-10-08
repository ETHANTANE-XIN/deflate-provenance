"""The three baselines of proposal section III.C.

1. **Brute-force zlib re-encoding** (the approach of early Precomp versions):
   decompress a stream and recompress it with CPython's zlib at every
   combination of compression level (1-9) and memory level (1-9) -- the same
   81 combinations as the team's preliminary check.  An exact byte match
   proves the stream is reproducible by that zlib build; the baseline can
   therefore only ever answer "zlib" or "no match".  Every matching
   combination is returned, never just the first, because several settings
   often give identical output.
2. **preflate-rs parameter estimate**: Microsoft's preflate-rs (v0.7.6, the
   version cited) estimates the original encoder's match-finding parameters
   and hash function.  ``tools/preflate-estimate`` is a small Rust wrapper
   around its public ``preflate_whole_deflate_stream`` call; the hash function
   it reports is mapped to a profile (zlib, zlib-ng, libdeflate, Chromium's
   CRC32C-hash zlib, miniz).
3. **Container metadata alone**, using the fields zipdetails shows (creating
   system and version, version needed, general-purpose flags including the
   compression option bits, extra-field IDs, timestamps, attributes), in the
   spirit of Um et al.: a Random Forest over those fields predicts which
   program wrote the archive.  It never looks at the compressed data, so a
   metadata rewrite misleads it by construction.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import zlib
from typing import NamedTuple

from .containers import ContainerReport, METHOD_DEFLATE
from .deflate import parse_stream
from .ml.forest import RandomForest

# --- 1. brute-force zlib -------------------------------------------------------

#: the bytes a stream ends with after Z_SYNC_FLUSH then Z_FINISH: an empty
#: stored block (LEN 0, NLEN 0xFFFF) and an empty fixed-code final block
SYNC_FLUSH_TAIL = b"\x00\x00\xff\xff\x03\x00"
_CHUNK = 1 << 16


class ZlibMatch(NamedTuple):
    """One zlib setting that reproduces a stream byte for byte."""

    level: int
    mem_level: int
    sync_flush: bool = False

    def describe(self) -> str:
        return (f"level {self.level} memLevel {self.mem_level}"
                + (" + sync flush" if self.sync_flush else ""))


def _zlib_reproduces(output: bytes, raw: bytes, level: int, mem_level: int,
                     sync_flush: bool) -> bool:
    """Compress ``output`` and compare with ``raw`` as the bytes appear.

    zlib's output with ``Z_NO_FLUSH`` does not depend on how the input is
    split, so the input is fed in chunks and the comparison stops at the first
    differing byte -- for a wrong setting that is almost always inside the
    first block, which makes the 81-setting search cheap even on large entries.
    """
    comp = zlib.compressobj(level, zlib.DEFLATED, -15, mem_level, zlib.Z_DEFAULT_STRATEGY)
    done = 0
    for start in range(0, len(output), _CHUNK):
        part = comp.compress(output[start : start + _CHUNK])
        if part:
            if raw[done : done + len(part)] != part:
                return False
            done += len(part)
    if sync_flush:
        tail = comp.flush(zlib.Z_SYNC_FLUSH) + comp.flush(zlib.Z_FINISH)
    else:
        tail = comp.flush()
    return raw[done:] == tail


def zlib_reencode_matches(
    raw: bytes, output: bytes | None = None, levels=range(1, 10), mem_levels=range(1, 10),
) -> list[ZlibMatch]:
    """Every zlib setting at which CPython zlib reproduces ``raw`` exactly.

    Tries every (level, memLevel) combination, and, when the stream ends the
    way a sync flush before finishing leaves it, the same combinations with a
    sync flush.  All matches are returned, because several settings often give
    identical output.
    """
    if output is None:
        rec = parse_stream(raw, strict=False)
        if rec.error:
            return []
        output = rec.output
    if raw is None or output is None:
        return []
    found = [ZlibMatch(lv, ml) for lv in levels for ml in mem_levels
             if _zlib_reproduces(output, raw, lv, ml, False)]
    if not found and raw.endswith(SYNC_FLUSH_TAIL):
        found = [ZlibMatch(lv, ml, True) for lv in levels for ml in mem_levels
                 if _zlib_reproduces(output, raw, lv, ml, True)]
    return found


def zlib_baseline_label(raw: bytes, output: bytes | None = None) -> str:
    return "zlib" if zlib_reencode_matches(raw, output) else "unknown"


# --- 2. preflate-rs ----------------------------------------------------------

_REPO_TOOL = (
    Path(__file__).resolve().parents[1]
    / "tools" / "preflate-estimate" / "target" / "release" / "preflate-estimate"
)

#: preflate-rs HashAlgorithm -> profile name used by this project
PREFLATE_HASH_PROFILE = {
    "Zlib": "zlib",
    "ZlibNG": "zlib-ng",
    "Libdeflate4": "libdeflate",
    "Libdeflate4Fast": "libdeflate",
    "Crc32cHash": "chromium-zlib",
    "MiniZFast": "miniz",
}


def preflate_binary() -> str | None:
    for cand in (os.environ.get("DFP_PREFLATE_ESTIMATE"), shutil.which("preflate-estimate"),
                 str(_REPO_TOOL)):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def preflate_estimate(raws: list[bytes]) -> list[dict]:
    """Run preflate-rs on each raw stream; one result dict per stream."""
    exe = preflate_binary()
    if exe is None:
        raise RuntimeError(
            "preflate-estimate not built: run `cargo build --release --manifest-path "
            "tools/preflate-estimate/Cargo.toml` (needs crates.io access)")
    out: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, raw in enumerate(raws):
            p = Path(td) / f"s{i}.bin"
            p.write_bytes(raw)
            paths.append(str(p))
        for start in range(0, len(paths), 200):
            chunk = paths[start : start + 200]
            text = subprocess.run([exe, *chunk], capture_output=True, text=True,
                                  timeout=1800).stdout
            lines = {ln.split("\t", 1)[0]: ln.split("\t") for ln in text.splitlines() if ln}
            for p in chunk:
                parts = lines.get(p)
                if not parts or parts[1] != "OK":
                    out.append({"ok": False, "label": "unknown",
                                "error": parts[2] if parts and len(parts) > 2 else "no output"})
                    continue
                params = parts[4]
                m = re.search(r"hash_algorithm: (\w+)", params)
                algo = m.group(1) if m else "None"
                mt = re.search(r"matching_type: (\w+)", params)
                out.append({
                    "ok": True,
                    "hash_algorithm": algo,
                    "matching_type": mt.group(1) if mt else None,
                    "compressed_size": int(parts[2]),
                    "correction_bytes": int(parts[3]),
                    "label": PREFLATE_HASH_PROFILE.get(algo, "unknown"),
                    "parameters": params,
                })
    return out


# --- 3. container metadata only ---------------------------------------------

_EXTRA_IDS = ["0x0001", "0x000A", "0x5455", "0x7875", "0xCAFE", "0x7075", "0x6375", "0x5855"]
METADATA_FEATURES = (
    ["host_system", "version_made_by", "version_needed_known", "flag_data_descriptor",
     "flag_utf8", "option_bits", "n_extra_ids", "extra_unknown",
     "dos_date_is_1980", "dos_seconds_odd_frac", "distinct_timestamps_frac",
     "external_attr_unix", "external_attr_zero", "internal_attr_text",
     "stored_frac", "comment_len", "prepended"]
    + [f"extra_{x}" for x in _EXTRA_IDS]
)


def metadata_features(report: ContainerReport) -> list[float]:
    """Numeric summary of the zipdetails-visible fields of one archive."""
    entries = report.entries or []
    deflated = [e for e in entries if e["method"] == METHOD_DEFLATE] or entries
    first = deflated[0] if deflated else {}
    md = report.metadata
    extra_names: set[str] = set()
    for s in report.streams:
        for x in s.metadata.get("extra_fields", []):
            extra_names.add(x["id"])
    first_meta = report.streams[0].metadata if report.streams else {}
    ts = [s.metadata.get("dos_time", "") for s in report.streams]
    f = {
        "host_system": float(first.get("version_made_by", 0) >> 8),
        "version_made_by": float(first.get("version_made_by", 0) & 0xFF),
        "version_needed_known": float(bool(first)),
        "flag_data_descriptor": float(bool(first.get("flags", 0) & 0x08)),
        "flag_utf8": float(bool(first.get("flags", 0) & 0x800)),
        "option_bits": float(first.get("option_bits", 0)),
        "n_extra_ids": float(len(extra_names)),
        "extra_unknown": float(any(x not in _EXTRA_IDS for x in extra_names)),
        "dos_date_is_1980": float(any(t.startswith("1980") for t in ts)),
        "dos_seconds_odd_frac": 0.0,
        "distinct_timestamps_frac": (len(set(ts)) / len(ts)) if ts else 0.0,
        "external_attr_unix": float((first_meta.get("external_attr", 0) >> 16) != 0),
        "external_attr_zero": float(first_meta.get("external_attr", 0) == 0),
        "internal_attr_text": float(first_meta.get("internal_attr", 0) & 1),
        "stored_frac": (sum(1 for e in entries if e["method"] == 0) / len(entries))
        if entries else 0.0,
        "comment_len": float(md.get("archive_comment_len", 0)),
        "prepended": float(bool(md.get("prepended_bytes"))),
    }
    for x in _EXTRA_IDS:
        f[f"extra_{x}"] = float(x in extra_names)
    return [f[k] for k in METADATA_FEATURES]


class MetadataBaseline:
    """Random Forest over container metadata only (no compressed data)."""

    def __init__(self, n_estimators: int = 60, seed: int = 1) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.classes: list[str] = []
        self.forest: RandomForest | None = None

    def fit(self, reports: list[ContainerReport], labels: list[str]) -> "MetadataBaseline":
        X = np.array([metadata_features(r) for r in reports])
        self.classes = sorted(set(labels))
        y = np.array([self.classes.index(l) for l in labels])
        self.forest = RandomForest(n_estimators=self.n_estimators, seed=self.seed,
                                   min_samples_split=2, max_features=len(METADATA_FEATURES))
        self.forest.fit(X, y, len(self.classes))
        return self

    def predict(self, reports: list[ContainerReport]) -> list[str]:
        X = np.array([metadata_features(r) for r in reports])
        p = self.forest.predict_proba(X)
        return [self.classes[i] for i in p.argmax(axis=1)]
