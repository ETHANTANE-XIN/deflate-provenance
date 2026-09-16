"""A self-contained pure-Python RFC 1951 DEFLATE encoder.

This is deliberately *not* zlib.  It is written from the specification so that
its encoder-freedom choices differ from every zlib-derived family, which makes
it a genuine held-out "unknown" for the open-set evaluation.  It also exposes a
hook that lets the covert channel steer specific choices without changing the
decompressed bytes.

The encoder supports:

* greedy matching (take the longest match at each position) and
* lazy matching (defer by one byte when the next position offers a longer
  match), the switch zlib makes at higher levels,

so the same code can emit two behaviourally distinct families, ``purepy/greedy``
and ``purepy/lazy``.

Blocks are emitted as dynamic-Huffman blocks with a fixed, configurable
uncompressed span so the block-splitting policy is our own and recognisably
different from zlib's token-buffer-driven splitting.  Output is a raw DEFLATE
stream and always inflates (verified against zlib in the test suite).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..deflate import (
    CODE_LENGTH_ORDER,
    DIST_BASE,
    DIST_EXTRA,
    LENGTH_BASE,
    LENGTH_EXTRA,
    MAX_MATCH,
    MIN_MATCH,
    WINDOW_SIZE,
)
from .base import Encoder, EncoderResult, register


# --- bit writer (LSB-first, matching the reader) --------------------------


class BitWriter:
    __slots__ = ("_out", "_acc", "_nbits")

    def __init__(self) -> None:
        self._out = bytearray()
        self._acc = 0
        self._nbits = 0

    def write_bits(self, value: int, count: int) -> None:
        self._acc |= (value & ((1 << count) - 1)) << self._nbits
        self._nbits += count
        while self._nbits >= 8:
            self._out.append(self._acc & 0xFF)
            self._acc >>= 8
            self._nbits -= 8

    def write_code(self, code: int, length: int) -> None:
        """Write a Huffman code MSB-first (its bits reversed into the stream)."""
        for i in range(length - 1, -1, -1):
            self.write_bits((code >> i) & 1, 1)

    def finish(self) -> bytes:
        if self._nbits:
            self._out.append(self._acc & 0xFF)
            self._acc = 0
            self._nbits = 0
        return bytes(self._out)


# --- canonical Huffman code construction ----------------------------------


def build_canonical_codes(lengths: list[int]) -> list[int]:
    """Assign canonical codes from a code-length array (RFC 1951 3.2.2)."""
    max_len = max(lengths) if lengths else 0
    bl_count = [0] * (max_len + 1)
    for length in lengths:
        if length:
            bl_count[length] += 1
    code = 0
    next_code = [0] * (max_len + 1)
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    codes = [0] * len(lengths)
    for symbol, length in enumerate(lengths):
        if length:
            codes[symbol] = next_code[length]
            next_code[length] += 1
    return codes


def package_merge(freqs: list[int], max_bits: int) -> list[int]:
    """Length-limited Huffman code lengths via the package-merge algorithm.

    Returns a code-length array bounded by ``max_bits`` (15 for DEFLATE).  A
    length-limited construction is what real encoders use, so this keeps our
    trees plausible rather than degenerate.
    """
    symbols = [(f, i) for i, f in enumerate(freqs) if f > 0]
    n = len(symbols)
    lengths = [0] * len(freqs)
    if n == 0:
        return lengths
    if n == 1:
        lengths[symbols[0][1]] = 1
        return lengths

    freqs_only = sorted(symbols)  # ascending by frequency
    # Package-merge: build coin lists.
    base = [(f, (idx,)) for f, idx in freqs_only]
    packages = list(base)
    for _ in range(max_bits - 1):
        packages.sort(key=lambda p: p[0])
        merged = []
        it = iter(packages)
        for a in it:
            try:
                b = next(it)
            except StopIteration:
                break
            merged.append((a[0] + b[0], a[1] + b[1]))
        packages = sorted(base + merged, key=lambda p: p[0])

    packages.sort(key=lambda p: p[0])
    counts = [0] * len(freqs)
    needed = 2 * n - 2
    for _, members in packages[:needed]:
        for idx in members:
            counts[idx] += 1
    for i, c in enumerate(counts):
        if freqs[i] > 0:
            lengths[i] = max(1, min(c, max_bits))
    # Guarantee at least two used symbols get a length (DEFLATE needs a code).
    used = [i for i in range(len(freqs)) if freqs[i] > 0]
    for i in used:
        if lengths[i] == 0:
            lengths[i] = 1
    return lengths


def _fix_lengths(freqs: list[int], max_bits: int = 15) -> list[int]:
    lengths = package_merge(freqs, max_bits)
    # Ensure exactly a valid (not over-subscribed) set; package-merge can be
    # slightly off, so repair by the classic Kraft adjustment.
    used = [(lengths[i], i) for i in range(len(lengths)) if lengths[i] > 0]
    if not used:
        return lengths
    if len(used) == 1:
        lengths[used[0][1]] = 1
        return lengths
    # cap and rebalance to satisfy sum(2^-len) <= 1
    for _ in range(64):
        kraft = sum(2 ** (max_bits - length) for length, _ in used)
        limit = 1 << max_bits
        if kraft <= limit:
            break
        used.sort()  # shortest first
        # lengthen the most frequent short code
        length, idx = used[0]
        if length < max_bits:
            lengths[idx] += 1
            used[0] = (length + 1, idx)
        else:
            break
    return lengths


# --- code-length alphabet encoding ----------------------------------------


def _encode_code_lengths(all_lengths: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    """Run-length encode a code-length array into CL symbols (with extra bits).

    Returns ``(cl_symbol_stream, [(symbol, extra_value)...])``.  We use a
    conservative policy (repeat 16/17/18 only for runs of 3+) that is distinct
    from zlib's, which is part of our fingerprint.
    """
    result: list[tuple[int, int]] = []
    i = 0
    n = len(all_lengths)
    while i < n:
        value = all_lengths[i]
        run = 1
        while i + run < n and all_lengths[i + run] == value:
            run += 1
        if value == 0:
            while run >= 11:
                take = min(run, 138)
                result.append((18, take - 11))
                run -= take
                i += take
            while run >= 3:
                take = min(run, 10)
                result.append((17, take - 3))
                run -= take
                i += take
            for _ in range(run):
                result.append((0, 0))
                i += 1
        else:
            result.append((value, 0))
            i += 1
            run -= 1
            while run >= 3:
                take = min(run, 6)
                result.append((16, take - 3))
                run -= take
                i += take
            for _ in range(run):
                result.append((value, 0))
                i += 1
    cl_symbols = [s for s, _ in result]
    return cl_symbols, result


# --- LZ77 match finder -----------------------------------------------------


@dataclass
class Token:
    literal: int = -1
    length: int = 0
    distance: int = 0

    @property
    def is_match(self) -> bool:
        return self.length >= MIN_MATCH


class PurePyEncoder(Encoder):
    family = "purepy"

    def __init__(self, block_span: int = 32768, max_chain: int = 128) -> None:
        self.block_span = block_span
        self.max_chain = max_chain

    def available(self) -> bool:
        return True

    def levels(self) -> list[str]:
        return ["greedy", "lazy"]

    # -- match finding ------------------------------------------------------
    def _find_match(
        self, data: bytes, pos: int, head: dict, chain: list[int]
    ) -> tuple[int, int]:
        n = len(data)
        if pos + MIN_MATCH > n:
            return 0, 0
        key = data[pos : pos + 3]
        cand = head.get(key, -1)
        best_len = 0
        best_dist = 0
        tries = self.max_chain
        max_len = min(MAX_MATCH, n - pos)
        while cand >= 0 and tries > 0:
            dist = pos - cand
            if dist > WINDOW_SIZE:
                break
            length = 0
            while length < max_len and data[cand + length] == data[pos + length]:
                length += 1
            if length > best_len:
                best_len = length
                best_dist = dist
                if length >= max_len:
                    break
            cand = chain[cand] if cand < len(chain) else -1
            tries -= 1
        if best_len >= MIN_MATCH:
            return best_len, best_dist
        return 0, 0

    def _tokenize(self, data: bytes, lazy: bool, steer=None) -> list[Token]:
        n = len(data)
        head: dict[bytes, int] = {}
        chain = [-1] * n
        tokens: list[Token] = []
        pos = 0

        def insert(p: int) -> None:
            if p + 3 <= n:
                key = data[p : p + 3]
                chain[p] = head.get(key, -1)
                head[key] = p

        while pos < n:
            length, dist = self._find_match(data, pos, head, chain)
            if lazy and length >= MIN_MATCH and pos + 1 < n:
                insert(pos)
                next_len, next_dist = self._find_match(data, pos + 1, head, chain)
                if next_len > length:
                    # defer: emit a literal, take the better match next round
                    tokens.append(Token(literal=data[pos]))
                    pos += 1
                    continue
                # commit current match; positions inside are inserted below
                if steer is not None:
                    length, dist = steer(pos, length, dist, data, head, chain)
                tokens.append(Token(length=length, distance=dist))
                for k in range(1, length):
                    insert(pos + k)
                pos += length
                continue
            if length >= MIN_MATCH:
                if steer is not None:
                    length, dist = steer(pos, length, dist, data, head, chain)
                tokens.append(Token(length=length, distance=dist))
                for k in range(length):
                    insert(pos + k)
                pos += length
            else:
                tokens.append(Token(literal=data[pos]))
                insert(pos)
                pos += 1
        return tokens

    # -- block emission -----------------------------------------------------
    def _emit_block(self, writer: BitWriter, tokens: list[Token], is_final: bool) -> None:
        lit_freq = [0] * 288
        dist_freq = [0] * 30
        lit_freq[256] = 1  # end-of-block
        for tok in tokens:
            if tok.is_match:
                length_index = _length_to_index(tok.length)
                lit_freq[257 + length_index] += 1
                dist_freq[_dist_to_index(tok.distance)] += 1
            else:
                lit_freq[tok.literal] += 1

        lit_lengths = _fix_lengths(lit_freq, 15)
        # DEFLATE requires HLIT >= 257 (codes 0..256 present in the array)
        hlit = max(257, _last_nonzero(lit_lengths) + 1)
        lit_lengths = lit_lengths[:hlit] + [0] * (hlit - len(lit_lengths[:hlit]))
        while len(lit_lengths) < hlit:
            lit_lengths.append(0)

        if any(dist_freq):
            dist_lengths = _fix_lengths(dist_freq, 15)
        else:
            dist_lengths = [1, 1]  # minimal legal distance tree
        hdist = max(1, _last_nonzero(dist_lengths) + 1)
        dist_lengths = dist_lengths[:hdist]
        while len(dist_lengths) < hdist:
            dist_lengths.append(0)
        if hdist == 1 and dist_lengths[0] == 0:
            dist_lengths[0] = 1

        lit_codes = build_canonical_codes(lit_lengths)
        dist_codes = build_canonical_codes(dist_lengths)

        combined = lit_lengths + dist_lengths
        cl_symbols, cl_stream = _encode_code_lengths(combined)
        cl_freq = [0] * 19
        for sym in cl_symbols:
            cl_freq[sym] += 1
        cl_lengths = _fix_lengths(cl_freq, 7)

        # HCLEN in the fixed transmit order
        order_lengths = [cl_lengths[CODE_LENGTH_ORDER[i]] for i in range(19)]
        hclen = 19
        while hclen > 4 and order_lengths[hclen - 1] == 0:
            hclen -= 1
        cl_codes = build_canonical_codes(cl_lengths)

        # header
        writer.write_bits(1 if is_final else 0, 1)
        writer.write_bits(2, 2)  # BTYPE = dynamic
        writer.write_bits(hlit - 257, 5)
        writer.write_bits(hdist - 1, 5)
        writer.write_bits(hclen - 4, 4)
        for i in range(hclen):
            writer.write_bits(order_lengths[i], 3)
        for sym, extra in cl_stream:
            writer.write_code(cl_codes[sym], cl_lengths[sym])
            if sym == 16:
                writer.write_bits(extra, 2)
            elif sym == 17:
                writer.write_bits(extra, 3)
            elif sym == 18:
                writer.write_bits(extra, 7)

        for tok in tokens:
            if tok.is_match:
                li = _length_to_index(tok.length)
                sym = 257 + li
                writer.write_code(lit_codes[sym], lit_lengths[sym])
                if LENGTH_EXTRA[li]:
                    writer.write_bits(tok.length - LENGTH_BASE[li], LENGTH_EXTRA[li])
                di = _dist_to_index(tok.distance)
                writer.write_code(dist_codes[di], dist_lengths[di])
                if DIST_EXTRA[di]:
                    writer.write_bits(tok.distance - DIST_BASE[di], DIST_EXTRA[di])
            else:
                writer.write_code(lit_codes[tok.literal], lit_lengths[tok.literal])
        writer.write_code(lit_codes[256], lit_lengths[256])

    def compress(self, data: bytes, level: str, steer=None) -> EncoderResult:
        lazy = level != "greedy"
        tokens = self._tokenize(data, lazy=lazy, steer=steer)
        writer = BitWriter()
        if not tokens:
            # empty input: one final empty dynamic block
            self._emit_block(writer, [], is_final=True)
        else:
            # split into blocks by output-span so the policy is our own
            blocks = self._split_blocks(tokens)
            for i, block in enumerate(blocks):
                self._emit_block(writer, block, is_final=(i == len(blocks) - 1))
        raw = writer.finish()
        return EncoderResult(
            raw_deflate=raw,
            family=self.family,
            level=level,
            label=self.label(level),
            extra={"lazy": lazy, "block_span": self.block_span},
        )

    def _split_blocks(self, tokens: list[Token]) -> list[list[Token]]:
        blocks: list[list[Token]] = []
        current: list[Token] = []
        span = 0
        for tok in tokens:
            current.append(tok)
            span += tok.length if tok.is_match else 1
            if span >= self.block_span:
                blocks.append(current)
                current = []
                span = 0
        if current:
            blocks.append(current)
        return blocks or [[]]


def _length_to_index(length: int) -> int:
    lo, hi = 0, 28
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if LENGTH_BASE[mid] <= length:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _dist_to_index(distance: int) -> int:
    lo, hi = 0, 29
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if DIST_BASE[mid] <= distance:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _last_nonzero(lengths: list[int]) -> int:
    for i in range(len(lengths) - 1, -1, -1):
        if lengths[i]:
            return i
    return 0


register(PurePyEncoder())
