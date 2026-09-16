"""Adversarial and anti-forensics analysis.

Three capabilities the deliverable calls for:

1. **Metadata-spoofing resistance.**  Traditional provenance reads container
   metadata (host system, timestamps, extra fields).  We show that rewriting
   all of that -- which :func:`normalise_zip_metadata` does -- destroys the
   metadata signal while leaving the bitstream feature vector untouched.  That
   invariance is the project's headline robustness result.

2. **Recompression detection.**  Decompress an entry and recompress it with a
   set of candidate encoders; if one candidate reproduces the original raw
   DEFLATE byte-for-byte, the file was (re)compressed by that encoder.  A near
   but inexact match with a *different* encoder than the metadata claims is
   evidence of recompression / rebuilding.

3. **Metadata vs bitstream comparison.**  A small helper that puts the two
   evidence channels side by side for the report's case studies.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from .containers import (
    ZIP_CENTRAL_SIG,
    ZIP_LOCAL_SIG,
    extract_streams,
)
from .deflate import parse_stream
from .encoders import list_encoders
from .features import extract_features, features_to_vector


# --- 1. metadata normalisation (the adversary's move) ---------------------


def normalise_zip_metadata(data: bytes) -> bytes:
    """Zero out the forgeable ZIP metadata, leaving compressed bytes intact.

    Rewrites version-made-by, general-purpose flags' informational bits, DOS
    timestamps and clears extra fields' *values* in both local and central
    headers.  The compressed payloads are copied verbatim, so every DEFLATE
    stream -- and therefore every bitstream feature -- is unchanged.

    This is intentionally a best-effort normaliser for the robustness
    experiment, not a general ZIP rewriter; it operates in place on a copy.
    """
    buf = bytearray(data)

    # local headers
    pos = 0
    while True:
        pos = buf.find(ZIP_LOCAL_SIG, pos)
        if pos < 0:
            break
        # version-need(4), flags(6), method(8), time(10), date(12)
        struct.pack_into("<H", buf, pos + 10, 0)  # mod time
        struct.pack_into("<H", buf, pos + 12, 0x0021)  # a fixed date
        name_len, extra_len = struct.unpack_from("<HH", buf, pos + 26)
        # neutralise extra field contents (keep length so offsets stay valid)
        extra_start = pos + 30 + name_len
        for i in range(extra_start, extra_start + extra_len):
            buf[i] = 0
        pos += 4

    # central directory headers
    pos = 0
    while True:
        pos = buf.find(ZIP_CENTRAL_SIG, pos)
        if pos < 0:
            break
        struct.pack_into("<H", buf, pos + 4, 20)  # version made by -> fixed
        struct.pack_into("<H", buf, pos + 12, 0)  # mod time
        struct.pack_into("<H", buf, pos + 14, 0x0021)  # date
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", buf, pos + 28)
        extra_start = pos + 46 + name_len
        for i in range(extra_start, extra_start + extra_len):
            buf[i] = 0
        pos += 4

    return bytes(buf)


@dataclass
class RobustnessResult:
    metadata_changed: bool
    bitstream_changed: bool
    n_streams: int
    max_feature_delta: float


def metadata_robustness(path: str) -> RobustnessResult:
    """Compare feature vectors before and after metadata normalisation."""
    original = extract_streams(path)
    orig_data = _read(path)
    scrubbed = normalise_zip_metadata(orig_data)
    scrubbed_report = extract_streams(path, data=scrubbed, container=original.container)

    max_delta = 0.0
    for a, b in zip(original.streams, scrubbed_report.streams):
        va = features_to_vector(extract_features(parse_stream(a.payload, strict=False)))
        vb = features_to_vector(extract_features(parse_stream(b.payload, strict=False)))
        delta = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
        max_delta = max(max_delta, delta)

    md_changed = original.metadata != scrubbed_report.metadata
    return RobustnessResult(
        metadata_changed=md_changed,
        bitstream_changed=max_delta > 1e-9,
        n_streams=len(original.streams),
        max_feature_delta=max_delta,
    )


# --- 2. recompression detection -------------------------------------------


@dataclass
class RecompressionMatch:
    matched: bool
    encoder: str | None
    exact: bool
    best_prefix_ratio: float
    note: str


def detect_recompression(raw_deflate: bytes) -> RecompressionMatch:
    """Try to reproduce a raw DEFLATE stream by recompressing its output.

    An exact byte match identifies the encoder+level that produced it.  This is
    the strongest possible attribution: it is a constructive proof, not a
    statistical guess.
    """
    try:
        output = zlib.decompress(raw_deflate, -15)
    except Exception as exc:
        return RecompressionMatch(False, None, False, 0.0, f"inflate failed: {exc}")

    best_ratio = 0.0
    for enc in list_encoders(only_available=True):
        for level in enc.levels():
            try:
                cand = enc.compress(output, level).raw_deflate
            except Exception:
                continue
            if cand == raw_deflate:
                return RecompressionMatch(
                    True, f"{enc.family}/{level}", True, 1.0,
                    "exact byte-for-byte reproduction",
                )
            # longest common prefix ratio as a closeness proxy
            n = min(len(cand), len(raw_deflate))
            common = 0
            for i in range(n):
                if cand[i] != raw_deflate[i]:
                    break
                common += 1
            ratio = common / max(len(raw_deflate), 1)
            best_ratio = max(best_ratio, ratio)
    return RecompressionMatch(
        False, None, False, best_ratio,
        "no exact reproduction; stream not produced by any known encoder/level "
        "in the panel (or a different version/build)",
    )


# --- 3. metadata vs bitstream side by side --------------------------------


def compare_channels(path: str, bitstream_label: str) -> dict:
    """Put the two independent evidence channels side by side for a report."""
    report = extract_streams(path)
    md = report.metadata
    claimed = md.get("host_systems") or []
    return {
        "container": report.container,
        "metadata_channel": {
            "host_systems": claimed,
            "version_made_by": md.get("version_made_by"),
            "first_timestamp": md.get("first_timestamp"),
            "extra_field_kinds": md.get("extra_field_kinds"),
            "note": "all of these fields are editable in seconds",
        },
        "bitstream_channel": {
            "attribution": bitstream_label,
            "note": "cannot be forged without re-running the original encoder",
        },
    }


def _read(path: str) -> bytes:
    from pathlib import Path

    return Path(path).read_bytes()
