"""A covert channel hidden in DEFLATE encoder freedom, and its detector.

The idea (the responsible anti-forensics half of the project): RFC 1951 leaves
the encoder free choices that do not affect the decompressed bytes.  We can
*steer* one such choice to carry hidden bits, so a file's payload is byte-for-
byte identical after decompression yet secretly encodes a message.

Carrier used here: **final-byte padding**.  After the last block, the stream is
padded to a byte boundary.  Those pad bits are ignored by every decompressor,
so we can place message bits there.  It is a low-capacity but perfectly
deniable carrier -- the decompressed output is unchanged and the file still
validates.

A second, higher-capacity carrier is exposed through the pure-Python encoder's
match-steering hook: at positions where two equal-length matches to different
distances exist, choosing the nearer or farther one encodes one bit without
changing the output.  This module implements the padding carrier end to end
(embed + extract + detect) and documents the match carrier, which the encoder
supports via its ``steer`` callback.

Detector: an innocent encoder zero-pads the final byte.  Non-zero, structured
padding across a corpus is anomalous.  The detector reports a per-stream flag
and a corpus-level chi-square-style score of how non-random the pad bits look.
"""

from __future__ import annotations

from dataclasses import dataclass

from .deflate import parse_stream


# --- padding carrier -------------------------------------------------------


def embed_in_padding(raw_deflate: bytes, message_bits: list[int]) -> bytes:
    """Overwrite the final byte's unused pad bits with message bits.

    Returns a new stream that decompresses identically.  Capacity is the number
    of pad bits (0..7); excess message bits are ignored, and the count actually
    written is recoverable by :func:`extract_from_padding` only if the receiver
    knows how many to expect (a length is normally prefixed by the caller).
    """
    record = parse_stream(raw_deflate, strict=False)
    pad_bits = record.final_pad_bits
    if pad_bits == 0 or not message_bits:
        return raw_deflate
    data = bytearray(raw_deflate)
    # After parsing, end_bit is byte-aligned (padding consumed). The byte that
    # carries the padding is the one just before that boundary; it holds
    # (8 - pad_bits) real stream bits, then pad_bits free bits.
    last_index = record.end_bit // 8 - 1
    used = 8 - pad_bits
    byte = data[last_index]
    mask = (1 << used) - 1
    byte &= mask
    for i, bit in enumerate(message_bits[:pad_bits]):
        if bit:
            byte |= 1 << (used + i)
    data[last_index] = byte
    return bytes(data)


def extract_from_padding(raw_deflate: bytes, n_bits: int) -> list[int]:
    """Recover up to ``n_bits`` message bits from the final-byte padding."""
    record = parse_stream(raw_deflate, strict=False)
    pad_bits = record.final_pad_bits
    if pad_bits == 0:
        return []
    last_index = record.end_bit // 8 - 1
    used = 8 - pad_bits
    byte = raw_deflate[last_index]
    out = []
    for i in range(min(n_bits, pad_bits)):
        out.append((byte >> (used + i)) & 1)
    return out


def bytes_to_bits(data: bytes) -> list[int]:
    return [(b >> i) & 1 for b in data for i in range(8)]


def bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        val = 0
        for j in range(8):
            val |= bits[i + j] << j
        out.append(val)
    return bytes(out)


def embed_message_across_streams(streams: list[bytes], message: bytes) -> list[bytes]:
    """Spread a message across many streams' padding (an archive-wide channel)."""
    bits = bytes_to_bits(message)
    result = []
    cursor = 0
    for raw in streams:
        record = parse_stream(raw, strict=False)
        cap = record.final_pad_bits
        chunk = bits[cursor : cursor + cap]
        cursor += len(chunk)
        result.append(embed_in_padding(raw, chunk) if chunk else raw)
    return result


# --- match-steering carrier (documented; uses the encoder hook) -----------


def make_bit_steerer(message_bits: list[int]):
    """Return a ``steer`` callback for :meth:`PurePyEncoder.compress`.

    At each match the callback consumes one message bit: bit 0 keeps the match,
    bit 1 shortens it by one byte when that is still a legal match (>= MIN_MATCH),
    trading a little ratio for a recoverable choice.  The decompressed bytes are
    unchanged because the encoder still covers the same input, just with a
    different token boundary.  This is a sketch of a higher-capacity carrier;
    the padding carrier above is the one fully verified in the tests.
    """
    state = {"i": 0}

    def steer(pos, length, distance, data, head, chain):
        if state["i"] >= len(message_bits):
            return length, distance
        bit = message_bits[state["i"]]
        state["i"] += 1
        if bit and length - 1 >= 3:
            return length - 1, distance
        return length, distance

    return steer


# --- detector --------------------------------------------------------------


@dataclass
class CovertFinding:
    stream: str
    pad_bits: int
    pad_value: int
    suspicious: bool
    reason: str


def detect_covert_padding(raw_deflate: bytes, name: str = "") -> CovertFinding:
    """Flag a single stream whose final padding is non-zero."""
    record = parse_stream(raw_deflate, strict=False)
    suspicious = record.final_pad_bits > 0 and record.final_pad_value != 0
    reason = (
        f"final {record.final_pad_bits} pad bits are non-zero "
        f"(0b{record.final_pad_value:0{max(record.final_pad_bits,1)}b})"
        if suspicious
        else "padding is zero (normal)"
    )
    return CovertFinding(
        stream=name,
        pad_bits=record.final_pad_bits,
        pad_value=record.final_pad_value,
        suspicious=suspicious,
        reason=reason,
    )


def corpus_padding_score(streams: list[bytes]) -> dict:
    """Corpus-level anomaly score: fraction of streams with non-zero padding.

    Innocent corpora sit near zero.  A channel that fills padding pushes the
    fraction toward the fraction of streams that have any pad bits at all, and
    makes the observed pad-value distribution look uniform rather than all-zero.
    """
    total = 0
    with_pad = 0
    nonzero = 0
    for raw in streams:
        rec = parse_stream(raw, strict=False)
        total += 1
        if rec.final_pad_bits:
            with_pad += 1
            if rec.final_pad_value:
                nonzero += 1
    frac_nonzero = nonzero / total if total else 0.0
    frac_of_padded = nonzero / with_pad if with_pad else 0.0
    return {
        "streams": total,
        "with_padding": with_pad,
        "nonzero_padding": nonzero,
        "fraction_nonzero": frac_nonzero,
        "fraction_of_padded_nonzero": frac_of_padded,
        "verdict": (
            "likely covert channel"
            if frac_of_padded > 0.30 and with_pad >= 8
            else "no padding-channel signal"
        ),
    }
