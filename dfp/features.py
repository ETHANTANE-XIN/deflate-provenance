"""Encoder-behaviour features derived from a parsed DEFLATE stream.

Each feature is a decision RFC 1951 left to the encoder.  Nothing here reads
container metadata, so the whole vector survives a metadata rewrite.

Groups
------
``blk_``   block-splitting policy: how many blocks, how large, which types
``cl_``    code-length alphabet: HCLEN and repeat-symbol (16/17/18) usage
``lit_``   literal/length Huffman tree shape
``dst_``   distance Huffman tree shape
``tok_``   LZ77 token mix, including the lazy-matching signature
``len_``   match-length distribution over the 29 length codes
``dis_``   match-distance distribution over the 30 distance codes
``str_``   stream-level artefacts: padding bits, density, block cadence

Those groups are *statistical* features: they describe the stream as a whole.
The *decision* features (``dec_`` and ``huf_``, see :mod:`dfp.decisions`)
compare each choice the encoder made with the alternatives that were
available -- longest match, deferred (lazy) matches, block boundaries and
block type, distance from the optimal Huffman table -- which the proposal
expects to depend on the encoder much more than on the content.

The vector is fixed-length and ordered; :data:`FEATURE_NAMES` is the canonical
order and is what a trained model stores alongside its parameters.  Always
obtain features through :func:`stream_features` (or parse with
``keep_tokens=True``), because the decision features need the token sequence.
"""

from __future__ import annotations

import math
from statistics import median

from .decisions import DECISION_FEATURE_NAMES, extract_decision_features
from .deflate import (
    BTYPE_DYNAMIC,
    BTYPE_STATIC,
    BTYPE_STORED,
    StreamRecord,
    parse_stream,
)

#: zlib emits stored blocks of exactly this size when level=0
ZLIB_STORE_BLOCK = 65535
#: zlib's default block accumulation target (deflate.c lit_bufsize at memLevel 8)
ZLIB_LIT_BUFSIZE = 16384


def _safe(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _entropy(counts: list[int] | list[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    acc = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            acc -= p * math.log2(p)
    return acc


def _gini(counts: list[int] | list[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def extract_features(record: StreamRecord) -> dict[str, float]:
    """Compute the ordered feature dictionary for one parsed stream."""
    f: dict[str, float] = {}
    all_blocks = record.blocks
    # Empty blocks (no output) are not choices about how to split content:
    # an empty stored block is a sync/full flush marker, an empty block at the
    # end is a terminator written after a flush (zlib, Word) or on close (Go).
    # The block-splitting features describe the blocks that carry data; the
    # flush and terminator habits get features of their own (section A2).
    blocks = [b for b in all_blocks if b.out_size > 0]
    n_blocks = len(blocks)
    out_size = record.out_size
    total_bits = record.total_bits
    n_tokens = record.n_tokens
    n_matches = record.n_matches

    # -- A. block-splitting policy -----------------------------------------
    stored = [b for b in blocks if b.btype == BTYPE_STORED]
    static = [b for b in blocks if b.btype == BTYPE_STATIC]
    dynamic = [b for b in blocks if b.btype == BTYPE_DYNAMIC]
    block_out = [float(b.out_size) for b in blocks]
    block_bits = [float(b.bit_length) for b in blocks]

    f["blk_count"] = float(n_blocks)
    f["blk_per_100kb"] = _safe(n_blocks * 100_000.0, out_size)
    f["blk_frac_stored"] = _safe(len(stored), n_blocks)
    f["blk_frac_static"] = _safe(len(static), n_blocks)
    f["blk_frac_dynamic"] = _safe(len(dynamic), n_blocks)
    f["blk_out_mean"] = _mean(block_out)
    f["blk_out_median"] = float(median(block_out)) if block_out else 0.0
    f["blk_out_max"] = max(block_out) if block_out else 0.0
    f["blk_out_min"] = min(block_out) if block_out else 0.0
    f["blk_out_cv"] = _safe(_stdev(block_out), _mean(block_out))
    f["blk_bits_mean"] = _mean(block_bits)
    f["blk_bits_cv"] = _safe(_stdev(block_bits), _mean(block_bits))
    f["blk_first_is_stored"] = float(bool(blocks) and blocks[0].btype == BTYPE_STORED)
    f["blk_first_is_static"] = float(bool(blocks) and blocks[0].btype == BTYPE_STATIC)
    f["blk_last_is_stored"] = float(bool(blocks) and blocks[-1].btype == BTYPE_STORED)
    f["blk_single_block"] = float(n_blocks == 1)
    # zlib level 0 signature: every full block exactly 65535 bytes
    full_store = [b for b in stored if b.stored_len == ZLIB_STORE_BLOCK]
    f["blk_store_65535_frac"] = _safe(len(full_store), max(len(stored), 1))
    f["blk_store_len_mean"] = _mean([float(b.stored_len or 0) for b in stored])
    f["blk_stored_pad_nonzero"] = float(any(b.stored_pad_value for b in stored))
    # cadence: are non-final blocks a constant uncompressed size?
    interior = block_out[:-1] if len(block_out) > 1 else []
    f["blk_interior_out_cv"] = _safe(_stdev(interior), _mean(interior))
    f["blk_interior_uniform"] = float(bool(interior) and _stdev(interior) < 1.0)
    # zlib flushes when its literal buffer fills; token counts then cluster
    tok_per_block = [float(b.n_tokens) for b in dynamic]
    f["blk_tokens_mean"] = _mean(tok_per_block)
    f["blk_tokens_cv"] = _safe(_stdev(tok_per_block), _mean(tok_per_block))
    f["blk_tokens_near_litbuf"] = _safe(
        sum(1 for t in tok_per_block if abs(t - ZLIB_LIT_BUFSIZE) <= 64),
        max(len(tok_per_block), 1),
    )

    # -- A2. flush markers and the end-of-stream terminator ------------------
    empty = [b for b in all_blocks if b.out_size == 0]
    last = all_blocks[-1] if all_blocks else None
    terminator = last if (last is not None and last.out_size == 0 and len(all_blocks) > 1) else None
    f["blk_flush_markers"] = float(sum(
        1 for b in empty if b.btype == BTYPE_STORED and b is not terminator))
    f["blk_empty_coded_nonfinal"] = float(sum(
        1 for b in empty if b.btype != BTYPE_STORED and b is not terminator))
    # Go's compress/flate closes with an empty *stored* final block; zlib after
    # a flush (and Microsoft Word) closes with an empty *fixed-code* final block
    f["blk_term_empty_stored"] = float(terminator is not None and terminator.btype == BTYPE_STORED)
    f["blk_term_empty_coded"] = float(terminator is not None and terminator.btype != BTYPE_STORED)

    # -- B. code-length alphabet -------------------------------------------
    hclens = [float(b.hclen or 0) for b in dynamic]
    hlits = [float(b.hlit or 0) for b in dynamic]
    hdists = [float(b.hdist or 0) for b in dynamic]
    cl_emitted = sum(b.cl_symbols_emitted for b in dynamic)
    f["cl_hclen_mean"] = _mean(hclens)
    f["cl_hclen_min"] = min(hclens) if hclens else 0.0
    f["cl_hclen_max"] = max(hclens) if hclens else 0.0
    f["cl_hclen_is_19_frac"] = _safe(
        sum(1 for h in hclens if h == 19), max(len(hclens), 1)
    )
    f["cl_hlit_mean"] = _mean(hlits)
    f["cl_hlit_is_286_frac"] = _safe(
        sum(1 for h in hlits if h == 286), max(len(hlits), 1)
    )
    f["cl_hlit_is_257_frac"] = _safe(
        sum(1 for h in hlits if h == 257), max(len(hlits), 1)
    )
    f["cl_hdist_mean"] = _mean(hdists)
    f["cl_hdist_is_1_frac"] = _safe(
        sum(1 for h in hdists if h <= 1), max(len(hdists), 1)
    )
    f["cl_hdist_is_30_frac"] = _safe(
        sum(1 for h in hdists if h == 30), max(len(hdists), 1)
    )
    f["cl_rep16_per_block"] = _safe(sum(b.cl_repeat_16 for b in dynamic), len(dynamic))
    f["cl_rep17_per_block"] = _safe(sum(b.cl_repeat_17 for b in dynamic), len(dynamic))
    f["cl_rep18_per_block"] = _safe(sum(b.cl_repeat_18 for b in dynamic), len(dynamic))
    f["cl_rep16_frac"] = _safe(sum(b.cl_repeat_16 for b in dynamic), cl_emitted)
    f["cl_rep17_frac"] = _safe(sum(b.cl_repeat_17 for b in dynamic), cl_emitted)
    f["cl_rep18_frac"] = _safe(sum(b.cl_repeat_18 for b in dynamic), cl_emitted)
    f["cl_uses_any_repeat"] = float(
        any(b.cl_repeat_16 or b.cl_repeat_17 or b.cl_repeat_18 for b in dynamic)
    )
    f["cl_symbols_per_block"] = _safe(cl_emitted, len(dynamic))
    f["cl_tree_bits_mean"] = _mean([float(b.tree_bits) for b in dynamic])
    f["cl_tree_bits_frac_of_stream"] = _safe(
        sum(b.tree_bits for b in dynamic), total_bits
    )
    # shape of the 19-entry code-length code-length array
    cl_arrays = [b.cl_code_lengths for b in dynamic if b.cl_code_lengths]
    if cl_arrays:
        first = cl_arrays[0]
        f["cl_arr_mean"] = _mean([float(v) for v in first])
        f["cl_arr_zeros_frac"] = _safe(sum(1 for v in first if v == 0), len(first))
        f["cl_arr_max"] = float(max(first))
        f["cl_arr_entropy"] = _entropy([float(v) for v in first])
    else:
        f["cl_arr_mean"] = 0.0
        f["cl_arr_zeros_frac"] = 0.0
        f["cl_arr_max"] = 0.0
        f["cl_arr_entropy"] = 0.0

    # -- C/D. Huffman tree shape -------------------------------------------
    lit_shape = [0.0] * 15
    dst_shape = [0.0] * 15
    for b in dynamic:
        for i, c in enumerate(b.literal_tree_shape[:15]):
            lit_shape[i] += c
        for i, c in enumerate(b.distance_tree_shape[:15]):
            dst_shape[i] += c
    lit_total = sum(lit_shape) or 1.0
    dst_total = sum(dst_shape) or 1.0
    for i in range(15):
        f[f"lit_shape_{i + 1}"] = lit_shape[i] / lit_total
    for i in range(15):
        f[f"dst_shape_{i + 1}"] = dst_shape[i] / dst_total
    f["lit_shape_entropy"] = _entropy(lit_shape)
    f["dst_shape_entropy"] = _entropy(dst_shape)
    f["lit_maxlen"] = float(
        max((i + 1 for i in range(15) if lit_shape[i]), default=0)
    )
    f["dst_maxlen"] = float(
        max((i + 1 for i in range(15) if dst_shape[i]), default=0)
    )
    f["lit_codes_used_mean"] = _mean(
        [float(sum(b.literal_tree_shape)) for b in dynamic]
    )
    f["dst_codes_used_mean"] = _mean(
        [float(sum(b.distance_tree_shape)) for b in dynamic]
    )
    f["lit_mean_code_len"] = _safe(
        sum((i + 1) * lit_shape[i] for i in range(15)), sum(lit_shape)
    )
    f["dst_mean_code_len"] = _safe(
        sum((i + 1) * dst_shape[i] for i in range(15)), sum(dst_shape)
    )

    # -- E. token mix and the lazy-match signature -------------------------
    f["tok_count"] = float(n_tokens)
    f["tok_literal_ratio"] = _safe(record.n_literals, n_tokens)
    f["tok_match_ratio"] = _safe(n_matches, n_tokens)
    f["tok_per_out_byte"] = _safe(n_tokens, out_size)
    f["tok_match_len_mean"] = _safe(record.match_len_sum, n_matches)
    f["tok_match_len_max"] = float(record.match_len_max)
    f["tok_len3_frac"] = _safe(record.len3_count, n_matches)
    f["tok_len258_frac"] = _safe(record.len258_count, n_matches)
    f["tok_dist1_frac"] = _safe(record.dist1_count, n_matches)
    f["tok_dist_max"] = float(record.dist_max)
    f["tok_log2dist_mean"] = _safe(record.log2dist_sum, n_matches)
    f["tok_coverage_by_matches"] = _safe(record.match_len_sum, out_size)
    # Lazy matching emits a literal then a match one position later, so runs of
    # exactly one literal between matches are its signature.  Greedy encoders
    # (and HUFFMAN_ONLY/RLE strategies) show a very different profile.
    f["tok_lit_runs_per_match"] = _safe(record.literal_runs, n_matches)
    f["tok_lit_run_mean"] = _safe(record.literal_run_len_sum, record.literal_runs)
    f["tok_lit_run_max"] = float(record.literal_run_len_max)
    f["tok_run1_frac"] = _safe(record.run1_literal_runs, max(record.literal_runs, 1))
    f["tok_lazy_signature"] = _safe(record.run1_literal_runs, max(n_matches, 1))
    f["tok_match_after_match_frac"] = _safe(record.match_after_match, max(n_matches, 1))

    # -- F. length and distance distributions ------------------------------
    len_hist = [float(v) for v in record.length_code_hist]
    dis_hist = [float(v) for v in record.dist_code_hist]
    len_total = sum(len_hist) or 1.0
    dis_total = sum(dis_hist) or 1.0
    for i in range(29):
        f[f"len_code_{i}"] = len_hist[i] / len_total
    for i in range(30):
        f[f"dis_code_{i}"] = dis_hist[i] / dis_total
    f["len_entropy"] = _entropy(len_hist)
    f["dis_entropy"] = _entropy(dis_hist)
    f["len_gini"] = _gini(len_hist)
    f["dis_gini"] = _gini(dis_hist)
    # window-reach features: how deep the encoder searched (distance code 24
    # starts at 4097, code 26 at 8193, code 28 at 16385; RFC 1951 3.2.5)
    f["dis_frac_gt_4k"] = _safe(sum(dis_hist[24:]), dis_total)
    f["dis_frac_gt_8k"] = _safe(sum(dis_hist[26:]), dis_total)
    f["dis_frac_gt_16k"] = _safe(sum(dis_hist[28:]), dis_total)
    f["dis_frac_le_256"] = _safe(sum(dis_hist[:16]), dis_total)
    f["dis_top_code"] = float(
        max(range(30), key=lambda i: dis_hist[i]) if sum(dis_hist) else 0
    )
    f["len_top_code"] = float(
        max(range(29), key=lambda i: len_hist[i]) if sum(len_hist) else 0
    )

    # -- G. stream-level ----------------------------------------------------
    f["str_final_pad_bits"] = float(record.final_pad_bits)
    f["str_final_pad_nonzero"] = float(record.final_pad_value != 0)
    f["str_bits_per_out_byte"] = _safe(total_bits, out_size)
    f["str_bits_per_token"] = _safe(total_bits, n_tokens)
    f["str_out_size_log"] = math.log10(out_size + 1.0)
    f["str_compressed_size_log"] = math.log10(record.compressed_bytes + 1.0)
    f["str_ratio"] = _safe(record.compressed_bytes, out_size)
    f["str_truncated"] = float(record.truncated)
    f["str_has_error"] = float(record.error is not None)

    # -- H. decision features ------------------------------------------------
    f.update(extract_decision_features(record))
    return f


def _empty_features() -> dict[str, float]:
    return extract_features(StreamRecord())


#: Canonical feature order.  Derived from an empty record so it can never drift
#: from what :func:`extract_features` actually produces.
FEATURE_NAMES: list[str] = list(_empty_features().keys())
N_FEATURES = len(FEATURE_NAMES)


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Project a feature dict onto :data:`FEATURE_NAMES` order."""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def vector_from_record(record: StreamRecord) -> list[float]:
    return features_to_vector(extract_features(record))


def stream_features(
    payload: bytes, start_bit: int = 0, max_bytes: int | None = None
) -> tuple[StreamRecord, dict[str, float]]:
    """Parse one raw DEFLATE stream and compute its full feature dictionary.

    This is the single entry point used by the corpus builder, the analyser
    and the adversarial tests, so training and analysis always see features
    computed the same way (tokens kept for the decision features).
    ``max_bytes`` caps how much of a very large stream is parsed (the analyser
    uses it; training streams are far below the cap).
    """
    record = parse_stream(payload, start_bit=start_bit, keep_tokens=True, strict=False,
                          max_bytes=max_bytes)
    features = extract_features(record)
    # tokens are only needed for the decision features; drop them to save memory
    for block in record.blocks:
        block.tokens = None
    return record, features


assert all(name in FEATURE_NAMES for name in DECISION_FEATURE_NAMES)
