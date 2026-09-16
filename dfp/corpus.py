"""Reproducible corpus generator.

The oracle is perfect because *we* make every file: for each synthetic content
sample we compress it with every available encoder and level, so the label
(which encoder produced this stream) is ground truth, not a guess.

Content is generated deterministically from a seed so a run is reproducible
without shipping any data.  Content types are chosen to stress different
encoder behaviours:

``text``    natural-language-like prose (many short matches, lazy matting)
``xml``     OOXML-like markup (long structural matches)
``source``  C-like source code (indentation runs, identifier repeats)
``json``    records with numeric fields
``csv``     tabular numeric data
``binary``  pseudo-random with embedded repeats (stresses match finding)
``rle``     long runs (stresses stored/RLE behaviour)

The generator yields :class:`Sample` rows; :func:`build_corpus` turns them into
an in-memory dataset of feature vectors + labels that the classifier trains on.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field

from .deflate import parse_stream
from .encoders import list_encoders
from .features import FEATURE_NAMES, features_to_vector, extract_features


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


@dataclass
class Sample:
    content_type: str
    size: int
    seed: int
    data: bytes


def generate_content(
    content_types: list[str] | None = None,
    sizes: list[int] | None = None,
    per_combo: int = 3,
    base_seed: int = 20260916,
) -> list[Sample]:
    """Produce deterministic content samples."""
    content_types = content_types or list(CONTENT_GENERATORS)
    sizes = sizes or [4_000, 20_000, 80_000]
    samples: list[Sample] = []
    counter = 0
    for ctype in content_types:
        gen = CONTENT_GENERATORS[ctype]
        for size in sizes:
            for k in range(per_combo):
                seed = base_seed + counter
                counter += 1
                rng = random.Random(seed)
                samples.append(Sample(ctype, size, seed, gen(rng, size)))
    return samples


@dataclass
class Dataset:
    X: list[list[float]] = field(default_factory=list)
    y: list[str] = field(default_factory=list)  # family label (the class)
    labels_full: list[str] = field(default_factory=list)  # family/level
    content_types: list[str] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    classes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)


def build_corpus(
    content_types: list[str] | None = None,
    sizes: list[int] | None = None,
    per_combo: int = 3,
    families: list[str] | None = None,
    label_by: str = "family",
    base_seed: int = 20260916,
    progress=None,
) -> Dataset:
    """Compress every content sample with every encoder; return features + labels.

    ``label_by`` is ``"family"`` (attribute to the implementation) or
    ``"family_level"`` (also distinguish the level).
    """
    samples = generate_content(content_types, sizes, per_combo, base_seed)
    encoders = list_encoders(only_available=True)
    if families:
        encoders = [e for e in encoders if e.family in families]

    ds = Dataset()
    total = len(samples) * sum(len(e.levels()) for e in encoders)
    done = 0
    for sample in samples:
        for enc in encoders:
            for level in enc.levels():
                try:
                    res = enc.compress(sample.data, level)
                    rec = parse_stream(res.raw_deflate, strict=False)
                except Exception:
                    done += 1
                    continue
                if rec.error:
                    done += 1
                    continue
                vec = features_to_vector(extract_features(rec))
                ds.X.append(vec)
                family_label = enc.family
                full = f"{enc.family}/{level}"
                ds.y.append(family_label if label_by == "family" else full)
                ds.labels_full.append(full)
                ds.content_types.append(sample.content_type)
                ds.sizes.append(sample.size)
                done += 1
                if progress and done % 50 == 0:
                    progress(done, total)
    ds.classes = sorted(set(ds.y))
    if progress:
        progress(total, total)
    return ds
