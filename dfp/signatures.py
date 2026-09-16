"""Deterministic signature rules.

These are transparent, hand-written rules that fire on unambiguous structural
evidence, independent of the statistical model.  A forensic report is stronger
when a machine-learned verdict is corroborated (or contradicted) by an
explainable rule, so every rule states plainly what it observed.

A rule returns a :class:`SignatureHit` or ``None``.  Rules never *attribute* on
their own beyond what is certain; most narrow the field or flag an anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .deflate import BTYPE_STORED, StreamRecord
from .features import extract_features


@dataclass
class SignatureHit:
    rule: str
    observation: str
    implication: str
    strength: str  # "strong" | "moderate" | "weak"


def rule_no_compression(record: StreamRecord) -> SignatureHit | None:
    stored = [b for b in record.blocks if b.btype == BTYPE_STORED]
    if record.blocks and len(stored) == len(record.blocks):
        full = [b for b in stored if b.stored_len == 65535]
        if full:
            return SignatureHit(
                "all-stored-65535",
                f"all {len(record.blocks)} blocks stored, "
                f"{len(full)} of exactly 65535 bytes",
                "consistent with zlib level 0 (no compression); "
                "rules out any lazy-matching encoder",
                "strong",
            )
        return SignatureHit(
            "all-stored",
            f"all {len(record.blocks)} blocks stored (uncompressed)",
            "level-0 / store-only encoder",
            "moderate",
        )
    return None


def rule_static_only(record: StreamRecord) -> SignatureHit | None:
    from .deflate import BTYPE_STATIC

    static = [b for b in record.blocks if b.btype == BTYPE_STATIC]
    if record.blocks and len(static) == len(record.blocks) and record.n_matches:
        return SignatureHit(
            "static-huffman-only",
            "every block uses fixed (static) Huffman codes",
            "consistent with zlib Z_FIXED strategy or a minimal encoder; "
            "excludes dynamic-tree encoders",
            "strong",
        )
    return None


def rule_no_matches(record: StreamRecord) -> SignatureHit | None:
    if record.n_tokens and record.n_matches == 0 and record.out_size > 64:
        return SignatureHit(
            "no-lz77-matches",
            "no back-references emitted despite compressible size",
            "consistent with Z_HUFFMAN_ONLY strategy (Huffman coding, no LZ77)",
            "strong",
        )
    return None


def rule_rle_only(record: StreamRecord) -> SignatureHit | None:
    if record.n_matches:
        near = record.dist1_count / record.n_matches
        if near > 0.98 and record.n_matches > 8:
            return SignatureHit(
                "distance-1-only",
                f"{near:.0%} of matches are distance 1 (run-length)",
                "consistent with Z_RLE strategy (match distance limited to 1)",
                "strong",
            )
    return None


def rule_single_dynamic(record: StreamRecord) -> SignatureHit | None:
    from .deflate import BTYPE_DYNAMIC

    if (
        len(record.blocks) == 1
        and record.blocks[0].btype == BTYPE_DYNAMIC
        and record.out_size > 200000
    ):
        return SignatureHit(
            "single-large-dynamic-block",
            f"one dynamic block spanning {record.out_size} bytes without a split",
            "encoder did not split a large stream; unusual for zlib's "
            "token-buffer flushing, seen in some one-shot encoders",
            "moderate",
        )
    return None


def rule_nonzero_padding(record: StreamRecord) -> SignatureHit | None:
    if record.final_pad_value:
        return SignatureHit(
            "nonzero-final-padding",
            f"final byte padded with non-zero bits (0b{record.final_pad_value:b})",
            "most encoders zero-pad; non-zero padding narrows the field and can "
            "carry a covert channel (see covert-channel detector)",
            "moderate",
        )
    return None


ALL_RULES = [
    rule_no_compression,
    rule_static_only,
    rule_no_matches,
    rule_rle_only,
    rule_single_dynamic,
    rule_nonzero_padding,
]


def evaluate_signatures(record: StreamRecord) -> list[SignatureHit]:
    hits = []
    for rule in ALL_RULES:
        try:
            hit = rule(record)
        except Exception:
            hit = None
        if hit:
            hits.append(hit)
    return hits
