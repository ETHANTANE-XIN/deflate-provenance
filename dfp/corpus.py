"""Reference corpus generator (proposal component 4).

The oracle is perfect because *we* make every file: each source file is
compressed with every available encoder and setting, so the label of every
stream is ground truth, not a guess.

What the builder guarantees (proposal sections III.B and III.C):

* **Sources are tracked.**  Every stream remembers the source file it was made
  from, so training and test sets can be split by source file and the same
  content never appears in both.  Synthetic sources get a unique seed per
  (content type, size, index), so no two sources share generated content.
* **Identical outputs share one label set.**  When several encoders or
  settings produce byte-identical output for a source (zlib levels 8 and 9 on
  some inputs; Java and CPython on the same zlib build), the stream is stored
  once and labelled with the whole set; predicting any member counts as
  correct.
* **Programs share a profile only when verified.**  An adapter that declares
  it is built on another library (Java on zlib, bsdtar on zlib, .NET on zlib
  or zlib-ng) is merged into that library's profile only if its output is
  byte-identical to the library's reference encoder on at least
  ``SHARE_THRESHOLD`` of its streams; otherwise it becomes a profile of its
  own.  The measured rates are written to the manifest.
* **Versions are recorded.**  The manifest stores every encoder's exact
  version, the generation parameters and the platform, so the corpus can be
  regenerated (the ``Dockerfile`` pins the versions).

Sources are either generated deterministically from a seed (the default, so a
run is reproducible without shipping data) or read from a directory of real
files (``--sources DIR``), for example a Govdocs1 sample.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .encoders import list_encoders
from .features import FEATURE_NAMES, features_to_vector, stream_features

SHARE_THRESHOLD = 0.95

# --- deterministic synthetic content --------------------------------------

_WORDS = (
    "the of and to in a is that for it as with was his he be not by but have "
    "you which are on or her had at from this they she will would there their "
    "forensic evidence deflate stream encoder provenance bitstream huffman "
    "compression archive metadata analysis classifier confidence attribution"
).split()


def _text(rng: random.Random, size: int) -> bytes:
    out: list[str] = []
    n = 0
    while n < size:
        w = rng.choice(_WORDS)
        out.append(w)
        n += len(w) + 1
        if rng.random() < 0.06:
            out.append(".\n")
            n += 2
    return " ".join(out).encode("utf-8")[:size]


def _xml(rng: random.Random, size: int) -> bytes:
    tags = ["w:p", "w:r", "w:t", "w:pPr", "w:rPr", "w:tbl", "w:tr", "w:tc"]
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document>']
    n = 0
    while n < size:
        t = rng.choice(tags)
        body = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 8)))
        frag = f"<{t}>{body}</{t}>"
        out.append(frag)
        n += len(frag)
    out.append("</w:document>")
    return "".join(out).encode("utf-8")[:size]


def _source(rng: random.Random, size: int) -> bytes:
    idents = ["record", "stream", "encoder", "value", "index", "buffer", "count",
              "result", "length", "distance", "token", "block", "feature"]
    lines: list[str] = []
    n = 0
    while n < size:
        depth = rng.randint(0, 3)
        indent = "    " * depth
        kind = rng.random()
        if kind < 0.4:
            line = f"{indent}{rng.choice(idents)} = {rng.choice(idents)}({rng.randint(0,99)})"
        elif kind < 0.7:
            line = f"{indent}for {rng.choice(idents)} in range({rng.randint(1,50)}):"
        else:
            line = f"{indent}return {rng.choice(idents)} + {rng.choice(idents)}"
        lines.append(line)
        n += len(line) + 1
    return "\n".join(lines).encode("utf-8")[:size]


def _json(rng: random.Random, size: int) -> bytes:
    rows: list[str] = []
    n = 0
    i = 0
    while n < size:
        row = (
            '{"id": %d, "name": "%s", "score": %.3f, "active": %s}'
            % (i, rng.choice(_WORDS), rng.random() * 100, rng.choice(["true", "false"]))
        )
        rows.append(row)
        n += len(row)
        i += 1
    return ("[" + ",\n".join(rows) + "]").encode("utf-8")[:size]


def _csv(rng: random.Random, size: int) -> bytes:
    rows = ["id,category,value,ratio"]
    n = len(rows[0])
    i = 0
    while n < size:
        row = f"{i},{rng.choice(_WORDS)},{rng.randint(0, 100000)},{rng.random():.6f}"
        rows.append(row)
        n += len(row) + 1
        i += 1
    return "\n".join(rows).encode("utf-8")[:size]


def _binary(rng: random.Random, size: int) -> bytes:
    out = bytearray()
    motifs = [bytes(rng.randrange(256) for _ in range(rng.randint(4, 16))) for _ in range(20)]
    while len(out) < size:
        if rng.random() < 0.5:
            out += bytes(rng.randrange(256) for _ in range(rng.randint(8, 40)))
        else:
            out += rng.choice(motifs)
    return bytes(out[:size])


def _rle(rng: random.Random, size: int) -> bytes:
    out = bytearray()
    while len(out) < size:
        out += bytes([rng.randrange(256)]) * rng.randint(20, 400)
    return bytes(out[:size])


CONTENT_GENERATORS = {
    "text": _text,
    "xml": _xml,
    "source": _source,
    "json": _json,
    "csv": _csv,
    "binary": _binary,
    "rle": _rle,
}

DEFAULT_SIZES = [500, 1500, 4000, 12000, 32000, 96000]


@dataclass
class Source:
    """One source file: the unit the train/test split is made over."""

    id: str
    content_type: str
    data: bytes
    origin: str

    @property
    def size(self) -> int:
        return len(self.data)


def _seed(base_seed: int, ctype: str, size: int, k: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{ctype}:{size}:{k}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def synthetic_sources(
    content_types: list[str] | None = None,
    sizes: list[int] | None = None,
    per_combo: int = 4,
    base_seed: int = 20260916,
) -> list[Source]:
    """Deterministic generated sources, one unique seed each."""
    content_types = content_types or list(CONTENT_GENERATORS)
    sizes = sizes or list(DEFAULT_SIZES)
    out: list[Source] = []
    for ctype in content_types:
        gen = CONTENT_GENERATORS[ctype]
        for size in sizes:
            for k in range(per_combo):
                seed = _seed(base_seed, ctype, size, k)
                data = gen(random.Random(seed), size)
                out.append(Source(f"syn:{ctype}:{size}:{k}", ctype, data,
                                  f"synthetic seed {seed}"))
    return out


_COMPRESSED_MAGIC = (
    b"\x89PNG", b"\xff\xd8\xff", b"PK\x03\x04", b"7z\xbc\xaf", b"\xfd7zXZ",
    b"BZh", b"\x28\xb5\x2f\xfd", b"wOF2", b"wOFF", b"%PDF",
)


def directory_sources(
    directories: str | Path | list,
    limit: int | None = None,
    max_bytes: int = 256_000,
    min_bytes: int = 200,
    label: str = "file",
    seen: set[str] | None = None,
) -> list[Source]:
    """Real files as sources (for example a Govdocs1 sample).

    Files are taken in a deterministic but type-mixing order (sorted by a hash
    of their path), gzip files are decompressed first, and files that are
    already compressed (PNG, JPEG, ZIP, 7z, xz, ...) are skipped because their
    content is close to random.  Content is capped at ``max_bytes``.

    A source's id is its content digest (``file:<digest>``), so the same
    content found in two directories is one source and can never fall on both
    sides of a train/test split.  Pass the same ``seen`` set to several calls
    to deduplicate across them; ``limit`` applies to each call.  ``label`` is
    kept for compatibility and recorded in the origin only.
    """
    import gzip

    dirs = [directories] if isinstance(directories, (str, Path)) else list(directories)
    paths = []
    for d in dirs:
        paths += [p for p in Path(d).rglob("*") if p.is_file() and not p.is_symlink()]
    paths.sort(key=lambda p: hashlib.sha256(str(p).encode()).hexdigest())
    out: list[Source] = []
    seen = set() if seen is None else seen
    for path in paths:
        try:
            data = path.read_bytes()
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
        except Exception:
            continue
        if data.startswith(_COMPRESSED_MAGIC):
            continue
        data = data[:max_bytes]
        if len(data) < min_bytes:
            continue
        digest = hashlib.sha256(data).hexdigest()[:16]
        if digest in seen:
            continue
        seen.add(digest)
        suffix = path.suffix.lower().lstrip(".")
        ctype = suffix if suffix and suffix != "gz" else "file"
        out.append(Source(f"file:{digest}", ctype, data, str(path)))
        if limit and len(out) >= limit:
            break
    return out


# --- corpus ------------------------------------------------------------------


@dataclass
class Corpus:
    """Feature matrix plus one metadata row per unique stream."""

    X: np.ndarray
    rows: list[dict]
    manifest: dict
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def __len__(self) -> int:
        return len(self.rows)

    # -- views -----------------------------------------------------------
    def profiles(self) -> list[str]:
        return sorted({r["profile"] for r in self.rows if r["profile"] and not r["synthetic"]})

    def mask(self, fn) -> np.ndarray:
        return np.array([bool(fn(r)) for r in self.rows], dtype=bool)

    def subset(self, mask: np.ndarray) -> "Corpus":
        idx = np.where(mask)[0]
        return Corpus(self.X[idx], [self.rows[i] for i in idx], self.manifest,
                      list(self.feature_names))

    # -- persistence -----------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "features.npz", X=self.X)
        (d / "rows.json").write_text(json.dumps(self.rows), encoding="utf-8")
        manifest = dict(self.manifest, feature_names=self.feature_names)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "Corpus":
        d = Path(directory)
        X = np.load(d / "features.npz")["X"]
        rows = json.loads((d / "rows.json").read_text(encoding="utf-8"))
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        names = manifest.get("feature_names", list(FEATURE_NAMES))
        if names != list(FEATURE_NAMES):
            raise ValueError(
                "corpus was built with a different feature set; rebuild it "
                "with `dfp corpus`"
            )
        return cls(X, rows, manifest, names)


# --- building ----------------------------------------------------------------


def _compress_source(args) -> dict:
    """Worker: compress one source with every encoder, featurise unique streams."""
    source, encoder_names = args
    from .encoders import get_encoder

    outputs: dict[str, str] = {}
    streams: dict[str, dict] = {}
    raws: dict[str, bytes] = {}
    errors: list[str] = []
    for name in encoder_names:
        enc = get_encoder(name)
        try:
            produced = enc.compress_many(source.data, enc.settings())
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        for setting, raw in produced.items():
            h = hashlib.sha256(raw).hexdigest()
            outputs[f"{name}/{setting}"] = h
            raws[h] = raw
    for h, raw in raws.items():
        record, feats = stream_features(raw)
        if record.error or record.output != source.data:
            errors.append(f"stream {h[:10]} failed verification: {record.error}")
            continue
        streams[h] = {
            "vector": features_to_vector(feats),
            "compressed_size": len(raw),
        }
    return {"source": source, "outputs": outputs, "streams": streams, "errors": errors}


def _verify_sharing(results: list[dict], encoders) -> dict:
    """Measure whether each non-reference program reproduces its library."""
    refs = {e.library: e.name for e in encoders if e.reference}
    report: dict[str, dict] = {}
    for enc in encoders:
        if enc.reference:
            continue
        ref = refs.get(enc.library)
        same = total = 0
        if ref:
            for res in results:
                ref_hashes = {
                    h for lbl, h in res["outputs"].items() if lbl.split("/")[0] == ref
                }
                for lbl, h in res["outputs"].items():
                    if lbl.split("/")[0] != enc.name:
                        continue
                    total += 1
                    same += h in ref_hashes
        rate = same / total if total else 0.0
        shared = bool(ref) and total > 0 and rate >= SHARE_THRESHOLD
        report[enc.name] = {
            "declared_library": enc.library,
            "reference": ref,
            "identical_streams": same,
            "streams": total,
            "identical_rate": rate,
            "shares_profile": shared,
            "profile": enc.library if shared else enc.name,
        }
    return report


def _canonical_setting(program: str, setting: str, profile: str, ref_program: str | None) -> str:
    return setting if program == ref_program or program == profile else f"{program}:{setting}"


def build_corpus(
    sources: list[Source] | None = None,
    encoders: list[str] | None = None,
    workers: int | None = None,
    progress=None,
    params: dict | None = None,
) -> Corpus:
    """Compress every source with every encoder; return the labelled corpus."""
    sources = sources if sources is not None else synthetic_sources()
    encs = list_encoders(only_available=True)
    if encoders:
        encs = [e for e in encs if e.name in encoders]
    for e in encs:  # build helpers once, before forking workers
        e.version()
    names = [e.name for e in encs]

    jobs = [(s, names) for s in sources]
    results: list[dict] = []
    workers = workers or min(8, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for i, res in enumerate(pool.map(_compress_source, jobs, chunksize=1)):
                results.append(res)
                if progress:
                    progress(i + 1, len(jobs))
    else:
        for i, job in enumerate(jobs):
            results.append(_compress_source(job))
            if progress:
                progress(i + 1, len(jobs))

    sharing = _verify_sharing(results, encs)
    enc_by_name = {e.name: e for e in encs}
    refs = {e.library: e.name for e in encs if e.reference}

    def profile_of(program: str) -> str:
        if program in sharing:
            return sharing[program]["profile"]
        return enc_by_name[program].library

    X: list[list[float]] = []
    rows: list[dict] = []
    errors: list[str] = []
    for res in results:
        src: Source = res["source"]
        errors += [f"{src.id}: {e}" for e in res["errors"]]
        producers: dict[str, list[str]] = defaultdict(list)
        for lbl, h in res["outputs"].items():
            producers[h].append(lbl)
        for h, labels in producers.items():
            if h not in res["streams"]:
                continue
            programs = sorted({lbl.split("/")[0] for lbl in labels})
            profiles = sorted({profile_of(p) for p in programs})
            synthetic = all(enc_by_name[p].synthetic for p in programs)
            settings: dict[str, list[str]] = defaultdict(list)
            for lbl in labels:
                prog, setting = lbl.split("/", 1)
                prof = profile_of(prog)
                settings[prof].append(
                    _canonical_setting(prog, setting, prof, refs.get(prof)))
            st = res["streams"][h]
            X.append(st["vector"])
            rows.append({
                "source_id": src.id,
                "content_type": src.content_type,
                "out_size": src.size,
                "compressed_size": st["compressed_size"],
                "profile": profiles[0] if len(profiles) == 1 else None,
                "profiles": profiles,
                "settings": {k: sorted(set(v)) for k, v in settings.items()},
                "labels": sorted(labels),
                "synthetic": synthetic,
                "origin": "corpus",
                "split": None,
                "hash": h,
            })

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dfp_version": _dfp_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "encoders": [e.describe() for e in encs],
        "profile_sharing": sharing,
        "share_threshold": SHARE_THRESHOLD,
        "n_sources": len(sources),
        "sources": [{"id": s.id, "content_type": s.content_type, "size": s.size,
                     "origin": s.origin} for s in sources],
        "params": params or {},
        "errors": errors[:200],
        "app_profiles": {},
    }
    matrix = np.array(X, dtype=np.float64) if X else np.zeros((0, len(FEATURE_NAMES)))
    return Corpus(matrix, rows, manifest)


def regenerate_source(meta: dict) -> Source:
    """Rebuild a source from its manifest entry (synthetic seed or file path)."""
    sid, origin = meta["id"], meta["origin"]
    if sid.startswith("syn:"):
        _, ctype, size, _k = sid.split(":")
        seed = int(origin.rsplit(" ", 1)[1])
        data = CONTENT_GENERATORS[ctype](random.Random(seed), int(size))
        return Source(sid, ctype, data, origin)
    import gzip

    data = Path(origin).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    data = data[: meta.get("size") or None]
    return Source(sid, meta["content_type"], data, origin)


def regenerate_streams(corpus: Corpus, source_ids: set[str]) -> dict[str, bytes]:
    """Recompress the given sources with the manifest's encoders.

    Returns ``{stream hash: raw DEFLATE}`` for every corpus row of those
    sources, and raises if any stream cannot be reproduced byte for byte --
    which doubles as a check that the corpus is regenerable.
    """
    from .encoders import get_encoder

    wanted = {r["hash"] for r in corpus.rows if r["source_id"] in source_ids
              and r.get("origin", "corpus") == "corpus"}
    metas = {m["id"]: m for m in corpus.manifest["sources"]}
    out: dict[str, bytes] = {}
    programs = [e["program"] for e in corpus.manifest["encoders"]]
    for sid in sorted(source_ids):
        if sid not in metas:
            continue
        src = regenerate_source(metas[sid])
        for prog in programs:
            enc = get_encoder(prog)
            if not enc.available():
                continue
            for raw in enc.compress_many(src.data, enc.settings()).values():
                h = hashlib.sha256(raw).hexdigest()
                if h in wanted:
                    out[h] = raw
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"{len(missing)} corpus streams could not be regenerated "
                           "(an encoder version changed?)")
    return out


def _dfp_version() -> str:
    from . import __version__

    return __version__
