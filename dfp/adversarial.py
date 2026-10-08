"""Metadata-robustness check and the exact re-encoding test.

1. **Metadata robustness.**  Traditional provenance reads container metadata
   (host system, timestamps, extra fields, option bits).  Rewriting all of it
   with :func:`dfp.zipwriter.normalise_metadata` changes the metadata but not
   one bit of any DEFLATE stream, so the bitstream feature vector must not
   move.  :func:`metadata_robustness` measures that invariance.
2. **Exact re-encoding.**  :func:`reencode_panel` decompresses a stream and
   recompresses it with every encoder and setting in the panel (plus the 81
   zlib level/memory-level combinations).  An exact byte match proves that the
   panel encoder reproduces the stream; every match is returned, because
   several settings often produce identical output.  This is corroboration of
   *which encoder wrote the current bytes*; it says nothing about an earlier
   encoder, because DEFLATE is lossless and recompression erases that trace
   (proposal section III.A).

The producer-rewrite and mixed-encoder edit experiments live in
:mod:`dfp.realfiles` and :mod:`dfp.evaluate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .baselines import zlib_reencode_matches
from .containers import extract_streams
from .deflate import parse_stream
from .encoders import list_encoders
from .features import features_to_vector, stream_features
from .zipwriter import normalise_metadata


@dataclass
class RobustnessResult:
    metadata_changed: bool
    bitstream_changed: bool
    n_streams: int
    max_feature_delta: float


def metadata_robustness(path: str) -> RobustnessResult:
    """Compare feature vectors before and after metadata normalisation."""
    data = Path(path).read_bytes()
    original = extract_streams(path, data=data)
    scrubbed = extract_streams(path, data=normalise_metadata(data),
                               container=original.container)
    max_delta = 0.0
    for a, b in zip(original.streams, scrubbed.streams):
        va = features_to_vector(stream_features(a.payload)[1])
        vb = features_to_vector(stream_features(b.payload)[1])
        max_delta = max(max_delta, max((abs(x - y) for x, y in zip(va, vb)), default=0.0))
    changed = (original.metadata != scrubbed.metadata
               or [s.metadata for s in original.streams] != [s.metadata for s in scrubbed.streams])
    return RobustnessResult(
        metadata_changed=changed,
        bitstream_changed=max_delta > 1e-9 or len(original.streams) != len(scrubbed.streams),
        n_streams=len(original.streams),
        max_feature_delta=max_delta,
    )


@dataclass
class ReencodeResult:
    matches: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def matched(self) -> bool:
        return bool(self.matches)


def reencode_panel(raw: bytes) -> ReencodeResult:
    """Every panel encoder/setting that reproduces ``raw`` byte for byte."""
    rec = parse_stream(raw, strict=False)
    if rec.error:
        return ReencodeResult(error=f"stream does not decode: {rec.error}")
    raw = raw[: rec.compressed_bytes]
    output = rec.output
    matches = [f"zlib/{m.describe()}" for m in zlib_reencode_matches(raw, output)]
    for enc in list_encoders(only_available=True):
        if enc.name == "zlib":
            continue
        try:
            produced = enc.compress_many(output, enc.settings())
        except Exception:
            continue
        matches += [f"{enc.name}/{s}" for s, r in produced.items() if r == raw]
    return ReencodeResult(matches=matches)
