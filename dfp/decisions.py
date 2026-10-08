"""Decision features: the encoder's choices compared with the alternatives.

Statistical features (:mod:`dfp.features`) describe what the stream contains.
Decision features use the fact that, after decompression, the tool knows the
original bytes, so at each point it can ask what the encoder *could* have done
and compare that with what it *did* (proposal section III.A):

* **matching** -- did the encoder take the longest match available?  when it
  stopped short, at what length (zlib's ``nice_length``)?  did it pick the
  nearest copy of that match?
* **lazy matching** -- when it emitted a literal although a match was
  available, was the literal followed by a longer match (a deferred match),
  or was the match simply skipped?
* **block boundaries** -- how many symbols each block holds, whether that
  count is a fixed buffer size, whether a split actually paid for itself, and
  whether the block type chosen (stored / fixed / dynamic) was the cheapest;
* **Huffman tables** -- how far each dynamic block's transmitted table is from
  the optimal length-limited table for the symbols the block really contains,
  and how many unused symbols were still given a code.

Match analysis is bounded so it stays affordable on large inputs: at most
``MAX_POSITIONS`` token positions are examined (evenly spaced over the
stream), each with a hash-chain style search of at most ``CHAIN_LIMIT``
earlier occurrences inside the 32 KiB window.  "Longest available" therefore
means the longest match that search finds, or the encoder's own match when that
is longer.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import median

from .deflate import (
    BTYPE_DYNAMIC,
    BTYPE_STORED,
    DIST_EXTRA,
    LENGTH_EXTRA,
    MAX_MATCH,
    MIN_MATCH,
    WINDOW_SIZE,
    StreamRecord,
)
from .huffman import code_cost, limited_lengths

MAX_POSITIONS = 1200
CHAIN_LIMIT = 48
#: zlib discards length-3 matches farther than this (deflate.c TOO_FAR)
TOO_FAR = 4096

#: Fixed-code lengths of RFC 1951 3.2.6, by symbol
_STATIC_LIT = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
_STATIC_DST = [5] * 30

DECISION_FEATURE_NAMES = [
    # matching
    "dec_match_longest_frac",
    "dec_match_shortfall_mean",
    "dec_shortfall_len_median",
    "dec_match_nearest_frac",
    "dec_match_farther_log2_mean",
    "dec_min_match_len",
    "dec_len3_far_frac",
    # literals and lazy matching
    "dec_lit_avail_frac",
    "dec_lit_avail_len_mean",
    "dec_lazy_defer_frac",
    "dec_lazy_nogain_frac",
    "dec_missed_match_frac",
    "dec_avail3_far_frac",
    # block boundaries and block type
    "dec_blk_tokens_mode_frac",
    "dec_blk_tokens_mode_log2",
    "dec_blk_tokens_pow2_frac",
    "dec_blk_boundary_gain_mean",
    "dec_blk_boundary_paid_frac",
    "dec_blk_type_suboptimal_frac",
    "dec_blk_static_saving_mean",
    # Huffman tables
    "huf_lit_excess_mean",
    "huf_dst_excess_mean",
    "huf_lit_optimal_frac",
    "huf_unused_lit_mean",
    "huf_unused_dst_mean",
    "huf_eob_len_mean",
    "huf_lit_limit_hit_frac",
]


def _match_len(buf: bytes, older: int, pos: int, maxlen: int) -> int:
    """Length of the common run starting at ``older`` and ``pos`` (>= 3 known)."""
    if buf[older : older + maxlen] == buf[pos : pos + maxlen]:
        return maxlen
    lo, hi = MIN_MATCH, maxlen - 1
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if buf[older : older + mid] == buf[pos : pos + mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _search(buf: bytes, pos: int, want: int) -> tuple[int, int, int, int]:
    """Search earlier copies of the bytes at ``pos``.

    Returns ``(best_len, best_dist, nearest_ge_want, nearest3_dist)`` where
    ``nearest_ge_want`` is the smallest distance whose match reaches ``want``
    bytes (0 if none was found) and ``nearest3_dist`` the nearest distance with
    any match of at least 3 bytes.
    """
    n = len(buf)
    maxlen = min(MAX_MATCH, n - pos)
    if maxlen < MIN_MATCH:
        return 0, 0, 0, 0
    prefix = buf[pos : pos + MIN_MATCH]
    lo = max(0, pos - WINDOW_SIZE)
    end = pos + MIN_MATCH - 1  # candidates c satisfy c + 3 <= end  ->  c < pos
    best_len = best_dist = nearest_ok = nearest3 = 0
    tries = 0
    while tries < CHAIN_LIMIT:
        cand = buf.rfind(prefix, lo, end)
        if cand < 0:
            break
        tries += 1
        end = cand + MIN_MATCH - 1
        dist = pos - cand
        if not nearest3:
            nearest3 = dist
        need_len = want and not nearest_ok
        if (
            not need_len
            and best_len >= MIN_MATCH
            and best_len < maxlen
            and buf[cand + best_len] != buf[pos + best_len]
        ):
            continue  # cannot beat the current best (zlib's quick reject)
        length = _match_len(buf, cand, pos, maxlen)
        if length > best_len:
            best_len, best_dist = length, dist
        if want and not nearest_ok and length >= want:
            nearest_ok = dist
        if best_len == maxlen and (nearest_ok or not want):
            break
    return best_len, best_dist, nearest_ok, nearest3


def _token_positions(record: StreamRecord) -> list[tuple[int, int, int, int]]:
    """Flatten compressed blocks into ``(pos, length, distance, next_len)``.

    Literals have ``length == 1`` and ``distance == 0``; ``next_len`` is the
    length of the following token in the same block (0 at a block end).
    """
    out: list[tuple[int, int, int, int]] = []
    pos = 0
    for block in record.blocks:
        if block.btype == BTYPE_STORED:
            pos += block.stored_len or 0
            continue
        toks = block.tokens or []
        for i, (a, b) in enumerate(toks):
            nxt = 0
            if i + 1 < len(toks):
                na, nb = toks[i + 1]
                nxt = 1 if nb < 0 else na
            if b < 0:
                out.append((pos, 1, 0, nxt))
                pos += 1
            else:
                out.append((pos, a, b, nxt))
                pos += a
    return out


def _entropy_bits(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    acc = 0.0
    for c in counts:
        if c > 0:
            acc -= c * math.log2(c / total)
    return acc


def _block_symbol_freqs(block) -> tuple[list[int], list[int], int]:
    """Literal/length and distance symbol frequencies, plus extra-bit total."""
    lit = list(block.literal_hist) + [1] + list(block.length_code_hist)  # 256 = EOB
    lit += [0] * (286 - len(lit))
    dst = list(block.dist_code_hist)
    extra = sum(c * LENGTH_EXTRA[i] for i, c in enumerate(block.length_code_hist))
    extra += sum(c * DIST_EXTRA[i] for i, c in enumerate(block.dist_code_hist))
    return lit, dst, extra


def extract_decision_features(record: StreamRecord) -> dict[str, float]:
    """Compute :data:`DECISION_FEATURE_NAMES` for one parsed stream.

    The record must have been parsed with ``keep_tokens=True`` and
    ``keep_output=True``; otherwise every feature is zero.
    """
    f = {name: 0.0 for name in DECISION_FEATURE_NAMES}
    if record.output is None or not record.blocks:
        return f
    _matching_features(record, f)
    _block_features(record, f)
    _huffman_features(record, f)
    return f


def _matching_features(record: StreamRecord, f: dict[str, float]) -> None:
    if any(b.tokens is None for b in record.blocks if b.btype != BTYPE_STORED):
        return
    buf = bytes(record.output)
    toks = _token_positions(record)
    usable = [t for t in toks if t[0] + MIN_MATCH <= len(buf)]
    if not usable:
        return
    n = len(usable)
    if n <= MAX_POSITIONS:
        sample = usable
    else:  # evenly spaced over the whole stream, first and last included
        sample = [usable[(i * (n - 1)) // (MAX_POSITIONS - 1)] for i in range(MAX_POSITIONS)]

    n_match = longest = 0
    shortfall_sum = 0.0
    shortfall_lens: list[int] = []
    nearest_known = nearest_hits = 0
    farther_log = 0.0
    n_len3 = len3_far = 0
    n_lit = lit_avail = defer = nogain = missed = 0
    avail_len_sum = 0
    avail3 = avail3_far = 0
    min_len = 0

    for pos, length, dist, nxt in sample:
        if dist:  # a match
            n_match += 1
            if min_len == 0 or length < min_len:
                min_len = length
            if length == MIN_MATCH:
                n_len3 += 1
                if dist > TOO_FAR:
                    len3_far += 1
            best, _, nearest_ok, _ = _search(buf, pos, length)
            best = max(best, length)
            if length >= best:
                longest += 1
            else:
                shortfall_sum += (best - length) / best
                shortfall_lens.append(length)
            if nearest_ok:
                nearest_known += 1
                if dist <= nearest_ok:
                    nearest_hits += 1
                else:
                    farther_log += math.log2(dist / nearest_ok)
        else:  # a literal
            n_lit += 1
            best, _, _, nearest3 = _search(buf, pos, 0)
            if best >= MIN_MATCH:
                lit_avail += 1
                avail_len_sum += best
                if best == MIN_MATCH:
                    avail3 += 1
                    if nearest3 > TOO_FAR:
                        avail3_far += 1
                if nxt > best:
                    defer += 1
                elif nxt >= MIN_MATCH:
                    nogain += 1
                else:
                    missed += 1

    f["dec_match_longest_frac"] = longest / n_match if n_match else 0.0
    f["dec_match_shortfall_mean"] = shortfall_sum / n_match if n_match else 0.0
    f["dec_shortfall_len_median"] = (
        float(median(shortfall_lens)) / MAX_MATCH if shortfall_lens else 0.0
    )
    f["dec_match_nearest_frac"] = nearest_hits / nearest_known if nearest_known else 0.0
    f["dec_match_farther_log2_mean"] = farther_log / nearest_known if nearest_known else 0.0
    f["dec_min_match_len"] = float(min_len)
    f["dec_len3_far_frac"] = len3_far / n_len3 if n_len3 else 0.0
    f["dec_lit_avail_frac"] = lit_avail / n_lit if n_lit else 0.0
    f["dec_lit_avail_len_mean"] = avail_len_sum / lit_avail if lit_avail else 0.0
    f["dec_lazy_defer_frac"] = defer / lit_avail if lit_avail else 0.0
    f["dec_lazy_nogain_frac"] = nogain / lit_avail if lit_avail else 0.0
    f["dec_missed_match_frac"] = missed / lit_avail if lit_avail else 0.0
    f["dec_avail3_far_frac"] = avail3_far / avail3 if avail3 else 0.0


def _block_features(record: StreamRecord, f: dict[str, float]) -> None:
    # Empty blocks are flush markers and end-of-stream terminators, not choices
    # about where to split content; :mod:`dfp.features` describes them itself.
    blocks = [b for b in record.blocks if b.out_size > 0]
    coded = [b for b in blocks if b.btype != BTYPE_STORED]
    # symbols per block except the last block that carries data (EOB included,
    # as encoders count it); a flush or an empty terminator after it does not
    # make that last block an interior one
    last = blocks[-1] if blocks else None
    nonfinal = [b.n_tokens + 1 for b in coded if b is not last]
    if nonfinal:
        mode, mode_n = Counter(nonfinal).most_common(1)[0]
        f["dec_blk_tokens_mode_frac"] = mode_n / len(nonfinal)
        f["dec_blk_tokens_mode_log2"] = math.log2(mode)
        pow2 = {v for k in range(8, 19) for v in ((1 << k) - 1, 1 << k, (1 << k) + 1)}
        f["dec_blk_tokens_pow2_frac"] = sum(
            1 for t in nonfinal if t in pow2 or (t - 1) in pow2
        ) / len(nonfinal)

    # does each split pay for itself?  entropy of the merged pair versus the
    # two halves, compared with the header the second block had to send
    gains = []
    paid = 0
    pairs = 0
    for a, b in zip(blocks, blocks[1:]):
        if a.btype != BTYPE_DYNAMIC or b.btype != BTYPE_DYNAMIC:
            continue
        la, da, _ = _block_symbol_freqs(a)
        lb, db, _ = _block_symbol_freqs(b)
        merged_l = [x + y for x, y in zip(la, lb)]
        merged_d = [x + y for x, y in zip(da, db)]
        saving = (
            _entropy_bits(merged_l) + _entropy_bits(merged_d)
            - _entropy_bits(la) - _entropy_bits(da)
            - _entropy_bits(lb) - _entropy_bits(db)
        )
        n = a.n_tokens + b.n_tokens + 2
        gains.append(saving / n)
        pairs += 1
        if saving >= b.tree_bits:
            paid += 1
    if gains:
        f["dec_blk_boundary_gain_mean"] = sum(gains) / len(gains)
        f["dec_blk_boundary_paid_frac"] = paid / pairs

    # was the block type the cheapest of the three the encoder could send?
    suboptimal = 0
    savings = []
    typed = 0
    for b in blocks:
        if b.btype == BTYPE_STORED:
            continue
        lit, dst, extra = _block_symbol_freqs(b)
        static_bits = code_cost(lit[:288], _STATIC_LIT[: len(lit)]) + code_cost(
            dst, _STATIC_DST
        ) + extra
        stored_bits = 8 * (b.out_size + 4 + (b.out_size // 65535) * 5) + 7
        if b.btype == BTYPE_DYNAMIC:
            dyn_bits = (
                code_cost(lit, b.literal_lengths + [0] * (286 - len(b.literal_lengths)))
                + code_cost(dst, b.distance_lengths + [0] * (30 - len(b.distance_lengths)))
                + extra
                + b.tree_bits
            )
            actual = dyn_bits
            savings.append((static_bits - dyn_bits) / max(dyn_bits, 1))
        else:
            actual = static_bits
        typed += 1
        if min(static_bits, stored_bits) < actual - 8:
            suboptimal += 1
    if typed:
        f["dec_blk_type_suboptimal_frac"] = suboptimal / typed
    if savings:
        f["dec_blk_static_saving_mean"] = sum(savings) / len(savings)


def _huffman_features(record: StreamRecord, f: dict[str, float]) -> None:
    dynamic = [b for b in record.blocks if b.btype == BTYPE_DYNAMIC]
    if not dynamic:
        return
    lit_excess = []
    dst_excess = []
    optimal = 0
    unused_lit = []
    unused_dst = []
    eob = []
    limit_hit = 0
    for b in dynamic:
        lit, dst, _ = _block_symbol_freqs(b)
        ll = b.literal_lengths + [0] * (286 - len(b.literal_lengths))
        dl = b.distance_lengths + [0] * (30 - len(b.distance_lengths))
        opt_l = limited_lengths(lit, 15)
        actual_l = code_cost(lit, ll)
        best_l = code_cost(lit, opt_l)
        lit_excess.append((actual_l - best_l) / best_l if best_l else 0.0)
        if actual_l == best_l:
            optimal += 1
        if sum(dst) >= 2 and sum(1 for x in dst if x) >= 2:
            opt_d = limited_lengths(dst, 15)
            best_d = code_cost(dst, opt_d)
            dst_excess.append((code_cost(dst, dl) - best_d) / best_d if best_d else 0.0)
        unused_lit.append(sum(1 for s in range(286) if ll[s] and not lit[s]))
        unused_dst.append(sum(1 for s in range(30) if dl[s] and not dst[s]))
        eob.append(ll[256])
        if max(ll) >= 15:
            limit_hit += 1
    n = len(dynamic)
    f["huf_lit_excess_mean"] = sum(lit_excess) / n
    f["huf_dst_excess_mean"] = sum(dst_excess) / len(dst_excess) if dst_excess else 0.0
    f["huf_lit_optimal_frac"] = optimal / n
    f["huf_unused_lit_mean"] = sum(unused_lit) / n
    f["huf_unused_dst_mean"] = sum(unused_dst) / n
    f["huf_eob_len_mean"] = sum(eob) / n
    f["huf_lit_limit_hit_frac"] = limit_hit / n
