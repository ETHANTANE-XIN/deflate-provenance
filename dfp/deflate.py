"""Bit-level RFC 1951 DEFLATE parser.

This module decodes a DEFLATE bitstream from first principles.  ``zlib`` is
never used for parsing -- only, in the test suite, as an independent oracle to
check that our inflated output matches byte for byte.

What makes this a *forensic* parser rather than a decompressor is that it keeps
every choice the encoder was free to make and that RFC 1951 does not record
anywhere in the file:

* the sequence of block types (stored / static Huffman / dynamic Huffman)
* exact block boundaries in bits, and the uncompressed size of each block
* the code-length alphabet: HCLEN, the 19 code-length code lengths, and how
  often the encoder used the repeat symbols 16 / 17 / 18
* both Huffman trees as full code-length arrays, so tree *shape* is available
* the literal/match token sequence, and derived order statistics that expose
  greedy versus lazy matching
* the padding bits between the last block and the byte boundary

Everything is returned as plain dataclasses so the feature layer, the report
layer and the covert-channel layer all read the same record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bitreader import BitReader, BitStreamError

# --- RFC 1951 constant tables ---------------------------------------------

BTYPE_STORED = 0
BTYPE_STATIC = 1
BTYPE_DYNAMIC = 2
BTYPE_RESERVED = 3

BTYPE_NAMES = {
    BTYPE_STORED: "stored",
    BTYPE_STATIC: "static",
    BTYPE_DYNAMIC: "dynamic",
    BTYPE_RESERVED: "reserved",
}

#: Order in which the 19 code-length code lengths are transmitted (RFC 1951 3.2.7)
CODE_LENGTH_ORDER = (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)

#: Length codes 257..285 -> (base length, extra bits)
LENGTH_BASE = (
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
    67, 83, 99, 115, 131, 163, 195, 227, 258,
)
LENGTH_EXTRA = (
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
    4, 4, 4, 4, 5, 5, 5, 5, 0,
)

#: Distance codes 0..29 -> (base distance, extra bits)
DIST_BASE = (
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
)
DIST_EXTRA = (
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
)

MAX_MATCH = 258
MIN_MATCH = 3
WINDOW_SIZE = 32768
END_OF_BLOCK = 256


class DeflateError(Exception):
    """Raised when the bitstream violates RFC 1951."""


def _static_literal_lengths() -> list[int]:
    """Fixed literal/length code lengths of RFC 1951 section 3.2.6."""
    lengths = [8] * 144 + [9] * 112 + [7] * 24 + [8] * 8
    assert len(lengths) == 288
    return lengths


STATIC_LITERAL_LENGTHS = _static_literal_lengths()
STATIC_DISTANCE_LENGTHS = [5] * 30


class Huffman:
    """Canonical Huffman decoder built from a code-length array.

    Uses the counts/symbols formulation from zlib's ``puff`` reference: for
    each bit length we know how many codes exist and which symbols they map
    to, so decoding walks one bit at a time with no decode table to build.
    """

    __slots__ = ("counts", "symbols", "lengths", "max_length", "used", "_table")

    def __init__(self, lengths: list[int], kind: str | None = None) -> None:
        """Build the decoder.

        ``kind`` applies zlib's completeness rules (``inftrees.c``) to a code
        read from a dynamic block header: ``"codes"`` (the code-length code)
        must be complete; ``"lens"`` and ``"dists"`` may be incomplete only in
        the single-code case (one code of length 1), and ``"dists"`` may also
        be empty.  ``None`` skips the check, which the fixed tables of RFC 1951
        3.2.6 need (30 five-bit distance codes do not fill the code space).
        """
        self.lengths = lengths
        counts = [0] * 16
        for length in lengths:
            if length < 0 or length > 15:
                raise DeflateError(f"illegal code length {length}")
            counts[length] += 1
        self.counts = counts
        self.used = len(lengths) - counts[0]
        left = 1
        for length in range(1, 16):
            left <<= 1
            left -= counts[length]
            if left < 0:
                raise DeflateError("over-subscribed Huffman code")
        self.max_length = max((i for i in range(1, 16) if counts[i]), default=0)
        if kind is not None and left > 0 and self.max_length:
            if kind == "codes" or self.max_length != 1:
                raise DeflateError(f"incomplete Huffman code ({kind})")
        offsets = [0] * 16
        for length in range(1, 15):
            offsets[length + 1] = offsets[length] + counts[length]
        symbols = [0] * self.used
        for symbol, length in enumerate(lengths):
            if length:
                symbols[offsets[length]] = symbol
                offsets[length] += 1
        self.symbols = symbols
        self._table: list[int] | None = None

    def table(self) -> list[int]:
        """Lookup table indexed by the next ``max_length`` stream bits.

        DEFLATE stores Huffman codes MSB-first inside an LSB-first bit stream,
        so each code is bit-reversed and replicated across every table slot
        whose low bits match it.  An entry packs ``symbol << 4 | length``;
        ``-1`` marks a bit pattern that no code covers (legal only in the
        incomplete single-code case, and an error if the stream uses it).
        """
        if self._table is None:
            size = 1 << self.max_length if self.max_length else 1
            table = [-1] * size
            code = 0
            next_code = [0] * 16
            for length in range(1, 16):
                prev = self.counts[length - 1] if length > 1 else 0
                code = (code + prev) << 1
                next_code[length] = code
            for symbol, length in enumerate(self.lengths):
                if not length:
                    continue
                c = next_code[length]
                next_code[length] += 1
                rev = 0
                for _ in range(length):
                    rev = (rev << 1) | (c & 1)
                    c >>= 1
                entry = (symbol << 4) | length
                for k in range(rev, size, 1 << length):
                    table[k] = entry
            self._table = table
        return self._table

    @property
    def incomplete(self) -> bool:
        left = 1
        for length in range(1, 16):
            left <<= 1
            left -= self.counts[length]
        return left != 0

    def decode(self, reader: BitReader) -> int:
        code = first = index = 0
        counts = self.counts
        for length in range(1, 16):
            code |= reader.read_bit()
            count = counts[length]
            if code - first < count:
                return self.symbols[index + (code - first)]
            index += count
            first = (first + count) << 1
            code <<= 1
        raise DeflateError("invalid Huffman code (ran past 15 bits)")

    def code_length_histogram(self) -> list[int]:
        """Counts of codes at each length 1..15 -- the tree's shape."""
        return list(self.counts[1:16])


_STATIC_CACHE: tuple[Huffman, Huffman] | None = None


def _static_tables() -> tuple[Huffman, Huffman]:
    """The fixed Huffman tables of RFC 1951 3.2.6, built once."""
    global _STATIC_CACHE
    if _STATIC_CACHE is None:
        _STATIC_CACHE = (
            Huffman(STATIC_LITERAL_LENGTHS),
            Huffman(STATIC_DISTANCE_LENGTHS),
        )
    return _STATIC_CACHE


# --- records ---------------------------------------------------------------

_ZERO_256: tuple[int, ...] = (0,) * 256
_ZERO_29: tuple[int, ...] = (0,) * 29
_ZERO_30: tuple[int, ...] = (0,) * 30


@dataclass
class BlockRecord:
    """Everything observable about one DEFLATE block."""

    index: int
    bfinal: int
    btype: int
    start_bit: int
    end_bit: int = 0
    header_bits: int = 0

    # stored blocks
    stored_len: int | None = None
    stored_nlen_ok: bool | None = None
    stored_pad_bits: int = 0
    stored_pad_value: int = 0

    # dynamic-block tree description
    hlit: int | None = None
    hdist: int | None = None
    hclen: int | None = None
    cl_code_lengths: list[int] = field(default_factory=list)  # 19 entries, transmit order
    cl_symbol_counts: dict[int, int] = field(default_factory=dict)  # symbol -> uses
    cl_repeat_16: int = 0
    cl_repeat_17: int = 0
    cl_repeat_18: int = 0
    cl_symbols_emitted: int = 0
    literal_lengths: list[int] = field(default_factory=list)
    distance_lengths: list[int] = field(default_factory=list)
    literal_tree_shape: list[int] = field(default_factory=list)  # counts at len 1..15
    distance_tree_shape: list[int] = field(default_factory=list)
    tree_bits: int = 0  # bits spent describing the trees

    # token statistics
    n_literals: int = 0
    n_matches: int = 0
    out_size: int = 0
    # Shared read-only zeros until the block decodes a token: a stream of empty
    # blocks must not cost ~4 KB of histograms per block (a few bytes of input).
    literal_hist: list[int] | tuple[int, ...] = _ZERO_256
    length_code_hist: list[int] | tuple[int, ...] = _ZERO_29
    dist_code_hist: list[int] | tuple[int, ...] = _ZERO_30
    match_len_sum: int = 0
    match_len_min: int = 0
    match_len_max: int = 0
    len3_count: int = 0
    len258_count: int = 0
    dist1_count: int = 0
    dist_max: int = 0
    log2dist_sum: float = 0.0
    literal_runs: int = 0
    literal_run_len_sum: int = 0
    literal_run_len_max: int = 0
    run1_literal_runs: int = 0  # runs of exactly one literal -> lazy-match proxy
    match_after_match: int = 0
    tokens: list[tuple[int, int]] | None = None  # (literal, -1) or (length, distance)

    @property
    def n_tokens(self) -> int:
        return self.n_literals + self.n_matches

    @property
    def bit_length(self) -> int:
        return self.end_bit - self.start_bit

    @property
    def type_name(self) -> str:
        return BTYPE_NAMES.get(self.btype, "?")

    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "bfinal": self.bfinal,
            "btype": self.btype,
            "type": self.type_name,
            "start_bit": self.start_bit,
            "end_bit": self.end_bit,
            "bit_length": self.bit_length,
            "out_size": self.out_size,
            "n_literals": self.n_literals,
            "n_matches": self.n_matches,
        }
        if self.btype == BTYPE_STORED:
            d.update(
                stored_len=self.stored_len,
                stored_nlen_ok=self.stored_nlen_ok,
                stored_pad_bits=self.stored_pad_bits,
                stored_pad_value=self.stored_pad_value,
            )
        if self.btype == BTYPE_DYNAMIC:
            d.update(
                hlit=self.hlit,
                hdist=self.hdist,
                hclen=self.hclen,
                cl_code_lengths=self.cl_code_lengths,
                cl_repeat_16=self.cl_repeat_16,
                cl_repeat_17=self.cl_repeat_17,
                cl_repeat_18=self.cl_repeat_18,
                tree_bits=self.tree_bits,
                literal_tree_shape=self.literal_tree_shape,
                distance_tree_shape=self.distance_tree_shape,
            )
        return d


@dataclass
class StreamRecord:
    """A parsed DEFLATE stream: its blocks plus stream-level artefacts."""

    blocks: list[BlockRecord] = field(default_factory=list)
    total_bits: int = 0
    start_bit: int = 0
    end_bit: int = 0
    out_size: int = 0
    final_pad_bits: int = 0
    final_pad_value: int = 0
    trailing_bytes: int = 0
    output: bytes | None = None
    truncated: bool = False
    error: str | None = None
    #: parsing stopped at a block boundary after ``max_bytes`` (analysis cap);
    #: the record then describes a prefix of the stream
    capped: bool = False

    # stream-level token totals (cheap to keep, avoids re-walking blocks)
    n_literals: int = 0
    n_matches: int = 0
    length_code_hist: list[int] = field(default_factory=lambda: [0] * 29)
    dist_code_hist: list[int] = field(default_factory=lambda: [0] * 30)
    match_len_sum: int = 0
    match_len_max: int = 0
    len3_count: int = 0
    len258_count: int = 0
    dist1_count: int = 0
    dist_max: int = 0
    log2dist_sum: float = 0.0
    literal_runs: int = 0
    literal_run_len_sum: int = 0
    literal_run_len_max: int = 0
    run1_literal_runs: int = 0
    match_after_match: int = 0

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def n_tokens(self) -> int:
        return self.n_literals + self.n_matches

    @property
    def compressed_bytes(self) -> int:
        return (self.end_bit - self.start_bit + 7) // 8

    def block_type_counts(self) -> dict[str, int]:
        counts = {"stored": 0, "static": 0, "dynamic": 0}
        for block in self.blocks:
            counts[block.type_name] = counts.get(block.type_name, 0) + 1
        return counts

    def to_dict(self, include_blocks: bool = True) -> dict:
        d = {
            "n_blocks": self.n_blocks,
            "block_types": self.block_type_counts(),
            "total_bits": self.total_bits,
            "out_size": self.out_size,
            "n_literals": self.n_literals,
            "n_matches": self.n_matches,
            "final_pad_bits": self.final_pad_bits,
            "final_pad_value": self.final_pad_value,
            "trailing_bytes": self.trailing_bytes,
            "truncated": self.truncated,
            "error": self.error,
        }
        if include_blocks:
            d["blocks"] = [b.to_dict() for b in self.blocks]
        return d


# --- the parser -----------------------------------------------------------


def _read_dynamic_tables(
    reader: BitReader, block: BlockRecord
) -> tuple[Huffman, Huffman]:
    """Decode a dynamic block's tree description, recording every artefact."""
    tree_start = reader.bit_pos
    hlit = reader.read_bits(5) + 257
    hdist = reader.read_bits(5) + 1
    hclen = reader.read_bits(4) + 4
    block.hlit, block.hdist, block.hclen = hlit, hdist, hclen
    # RFC 1951 3.2.7 allows the 5-bit fields to encode up to 288/32, but only
    # 286 literal/length and 30 distance symbols exist; zlib rejects more.
    if hlit > 286 or hdist > 30:
        raise DeflateError(f"too many length or distance symbols ({hlit}/{hdist})")

    cl_lengths = [0] * 19
    transmitted: list[int] = []
    for i in range(hclen):
        value = reader.read_bits(3)
        transmitted.append(value)
        cl_lengths[CODE_LENGTH_ORDER[i]] = value
    block.cl_code_lengths = transmitted

    cl_table = Huffman(cl_lengths, "codes")

    lengths: list[int] = []
    counts: dict[int, int] = {}
    n_wanted = hlit + hdist
    while len(lengths) < n_wanted:
        symbol = cl_table.decode(reader)
        counts[symbol] = counts.get(symbol, 0) + 1
        block.cl_symbols_emitted += 1
        if symbol < 16:
            lengths.append(symbol)
        elif symbol == 16:
            if not lengths:
                raise DeflateError("repeat symbol 16 with no previous length")
            repeat = 3 + reader.read_bits(2)
            lengths.extend([lengths[-1]] * repeat)
            block.cl_repeat_16 += 1
        elif symbol == 17:
            repeat = 3 + reader.read_bits(3)
            lengths.extend([0] * repeat)
            block.cl_repeat_17 += 1
        elif symbol == 18:
            repeat = 11 + reader.read_bits(7)
            lengths.extend([0] * repeat)
            block.cl_repeat_18 += 1
        else:
            raise DeflateError(f"illegal code-length symbol {symbol}")
    if len(lengths) > n_wanted:
        raise DeflateError("code-length repeat overflowed the alphabet")

    block.cl_symbol_counts = counts
    literal_lengths = lengths[:hlit]
    distance_lengths = lengths[hlit:]
    block.literal_lengths = literal_lengths
    block.distance_lengths = distance_lengths

    literal_table = Huffman(literal_lengths, "lens")
    if literal_lengths[END_OF_BLOCK] == 0:
        raise DeflateError("end-of-block symbol has no code")
    distance_table = Huffman(distance_lengths, "dists")
    block.literal_tree_shape = literal_table.code_length_histogram()
    block.distance_tree_shape = distance_table.code_length_histogram()
    block.tree_bits = reader.bit_pos - tree_start
    return literal_table, distance_table


def _decode_block_body(
    reader: BitReader,
    block: BlockRecord,
    literal_table: Huffman,
    distance_table: Huffman,
    out: bytearray,
    keep_tokens: bool,
) -> None:
    """Decode tokens until end-of-block, updating stats and the output window.

    This is the hot loop, so it works on a local bit position and table-driven
    Huffman lookups instead of calling :class:`BitReader` per bit.  The reader
    is re-synchronised to the final position before returning.
    """
    from math import log2

    data = reader._data
    total_bits = len(data) * 8
    pos = reader.bit_pos
    lit_tab = literal_table.table()
    lit_bits = literal_table.max_length
    lit_mask = (1 << lit_bits) - 1
    dist_tab = distance_table.table() if distance_table.max_length else [-1]
    dist_bits = distance_table.max_length
    dist_mask = (1 << dist_bits) - 1
    from_bytes = int.from_bytes

    tokens: list[tuple[int, int]] | None = [] if keep_tokens else None
    lit_hist = block.literal_hist = [0] * 256
    literal_run = 0
    prev_was_match = False
    len_hist = block.length_code_hist = [0] * 29
    dist_hist = block.dist_code_hist = [0] * 30
    min_len = 0
    n_literals = 0

    def peek(at: int) -> int:
        byte = at >> 3
        return from_bytes(data[byte : byte + 4], "little") >> (at & 7)

    try:
        while True:
            entry = lit_tab[peek(pos) & lit_mask]
            if entry < 0:
                raise DeflateError("invalid literal/length Huffman code")
            pos += entry & 15
            if pos > total_bits:
                raise BitStreamError("end of bitstream while decoding a symbol")
            symbol = entry >> 4
            if symbol < 256:
                out.append(symbol)
                lit_hist[symbol] += 1
                n_literals += 1
                literal_run += 1
                prev_was_match = False
                if tokens is not None:
                    tokens.append((symbol, -1))
                continue
            if symbol == END_OF_BLOCK:
                break
            # a match
            length_index = symbol - 257
            if length_index >= 29:
                raise DeflateError(f"illegal length symbol {symbol}")
            extra = LENGTH_EXTRA[length_index]
            length = LENGTH_BASE[length_index]
            if extra:
                length += peek(pos) & ((1 << extra) - 1)
                pos += extra
            if not dist_bits:
                raise DeflateError("match in a block with no distance codes")
            entry = dist_tab[peek(pos) & dist_mask]
            if entry < 0:
                raise DeflateError("invalid distance Huffman code")
            pos += entry & 15
            dist_symbol = entry >> 4
            if dist_symbol >= 30:
                raise DeflateError(f"illegal distance symbol {dist_symbol}")
            extra = DIST_EXTRA[dist_symbol]
            distance = DIST_BASE[dist_symbol]
            if extra:
                distance += peek(pos) & ((1 << extra) - 1)
                pos += extra
            if pos > total_bits:
                raise BitStreamError("end of bitstream inside a match")
            if distance > len(out):
                raise DeflateError(
                    f"distance {distance} exceeds {len(out)} bytes of history"
                )
            # LZ77 copy, overlapping-safe
            start = len(out) - distance
            if distance >= length:
                out += out[start : start + length]
            else:
                reps = -(-length // distance)
                out += (out[start:] * reps)[:length]

            block.n_matches += 1
            len_hist[length_index] += 1
            dist_hist[dist_symbol] += 1
            block.match_len_sum += length
            if length == MIN_MATCH:
                block.len3_count += 1
            elif length == MAX_MATCH:
                block.len258_count += 1
            if distance == 1:
                block.dist1_count += 1
            if distance > block.dist_max:
                block.dist_max = distance
            if min_len == 0 or length < min_len:
                min_len = length
            if length > block.match_len_max:
                block.match_len_max = length
            block.log2dist_sum += log2(distance)
            if literal_run:
                block.literal_runs += 1
                block.literal_run_len_sum += literal_run
                if literal_run == 1:
                    block.run1_literal_runs += 1
                if literal_run > block.literal_run_len_max:
                    block.literal_run_len_max = literal_run
                literal_run = 0
            elif prev_was_match:
                block.match_after_match += 1
            prev_was_match = True
            if tokens is not None:
                tokens.append((length, distance))
    finally:
        block.n_literals += n_literals
        if not n_literals and not block.n_matches:
            block.literal_hist = _ZERO_256
            block.length_code_hist = _ZERO_29
            block.dist_code_hist = _ZERO_30
        if pos > total_bits:
            reader.seek_bit(total_bits)
        else:
            reader.seek_bit(pos)
    if pos > total_bits:
        raise BitStreamError("end of bitstream while decoding a block")

    if literal_run:
        block.literal_runs += 1
        block.literal_run_len_sum += literal_run
        if literal_run == 1:
            block.run1_literal_runs += 1
        if literal_run > block.literal_run_len_max:
            block.literal_run_len_max = literal_run
    block.match_len_min = min_len
    block.tokens = tokens


def inflate(
    data: bytes,
    start_bit: int = 0,
    keep_output: bool = True,
    keep_tokens: bool = False,
    max_blocks: int = 1_000_000,
    strict: bool = True,
    max_bytes: int | None = None,
) -> StreamRecord:
    """Parse a raw DEFLATE stream and return a :class:`StreamRecord`.

    Parameters
    ----------
    data:
        Buffer holding the stream (may contain trailing container bytes).
    start_bit:
        Absolute bit offset of the first block header.
    keep_output:
        Retain the decompressed bytes on the record (needed to verify the
        parser against an oracle, and to compute the true uncompressed size).
    keep_tokens:
        Retain the full token sequence per block.  Off by default: the
        aggregate statistics carry the forensic signal at a fraction of the
        memory.
    strict:
        Raise :class:`DeflateError` on malformed input.  When False, the error
        is recorded on the returned record and parsing stops -- the behaviour
        wanted for damaged evidence.
    max_bytes:
        Stop after the first block that ends beyond this many compressed bytes
        and mark the record ``capped`` (it then describes a prefix of the
        stream).  ``None`` parses everything.
    """
    reader = BitReader(data, start_bit)
    record = StreamRecord(start_bit=start_bit)
    out = bytearray()

    try:
        index = 0
        while True:
            if index >= max_blocks:
                raise DeflateError(f"block count exceeded {max_blocks}")
            block_start = reader.bit_pos
            bfinal = reader.read_bit()
            btype = reader.read_bits(2)
            block = BlockRecord(
                index=index,
                bfinal=bfinal,
                btype=btype,
                start_bit=block_start,
                header_bits=3,
            )
            out_before = len(out)

            if btype == BTYPE_RESERVED:
                raise DeflateError("reserved block type 3")
            if btype == BTYPE_STORED:
                pad_bits, pad_value = reader.align_to_byte()
                block.stored_pad_bits = pad_bits
                block.stored_pad_value = pad_value
                length = int.from_bytes(reader.read_aligned_bytes(2), "little")
                nlen = int.from_bytes(reader.read_aligned_bytes(2), "little")
                block.stored_len = length
                block.stored_nlen_ok = (nlen == (~length & 0xFFFF))
                if not block.stored_nlen_ok:
                    # Not a DEFLATE stream (or a damaged one): record and stop
                    # in non-strict mode, as zlib does, instead of copying LEN
                    # bytes of whatever follows and classifying them.
                    raise DeflateError(f"stored block LEN/NLEN mismatch ({length}/{nlen})")
                out += reader.read_aligned_bytes(length)
                block.n_literals = length
            elif btype == BTYPE_STATIC:
                literal_table, distance_table = _static_tables()
                block.literal_tree_shape = literal_table.code_length_histogram()
                block.distance_tree_shape = distance_table.code_length_histogram()
                _decode_block_body(
                    reader, block, literal_table, distance_table, out, keep_tokens
                )
            else:
                literal_table, distance_table = _read_dynamic_tables(reader, block)
                _decode_block_body(
                    reader, block, literal_table, distance_table, out, keep_tokens
                )

            block.end_bit = reader.bit_pos
            block.out_size = len(out) - out_before
            record.blocks.append(block)
            _accumulate(record, block)
            index += 1
            if bfinal:
                break
            if max_bytes is not None and (reader.bit_pos - start_bit) // 8 >= max_bytes:
                record.capped = True
                break
    except (DeflateError, BitStreamError) as exc:
        record.truncated = isinstance(exc, BitStreamError)
        record.error = str(exc)
        if strict:
            record.end_bit = reader.bit_pos
            record.total_bits = record.end_bit - start_bit
            record.out_size = len(out)
            if keep_output:
                record.output = bytes(out)
            raise DeflateError(str(exc)) from exc

    record.end_bit = reader.bit_pos
    record.total_bits = record.end_bit - start_bit
    record.out_size = len(out)
    if keep_output:
        record.output = bytes(out)
    if record.error is None and not record.capped:
        pad_bits, pad_value = reader.align_to_byte()
        record.final_pad_bits = pad_bits
        record.final_pad_value = pad_value
        record.end_bit = reader.bit_pos
        record.trailing_bytes = len(data) - (reader.bit_pos // 8)
    return record


def _accumulate(record: StreamRecord, block: BlockRecord) -> None:
    """Roll a block's statistics into the stream totals."""
    record.n_literals += block.n_literals
    record.n_matches += block.n_matches
    for i in range(29):
        record.length_code_hist[i] += block.length_code_hist[i]
    for i in range(30):
        record.dist_code_hist[i] += block.dist_code_hist[i]
    record.match_len_sum += block.match_len_sum
    record.match_len_max = max(record.match_len_max, block.match_len_max)
    record.len3_count += block.len3_count
    record.len258_count += block.len258_count
    record.dist1_count += block.dist1_count
    record.dist_max = max(record.dist_max, block.dist_max)
    record.log2dist_sum += block.log2dist_sum
    record.literal_runs += block.literal_runs
    record.literal_run_len_sum += block.literal_run_len_sum
    record.literal_run_len_max = max(
        record.literal_run_len_max, block.literal_run_len_max
    )
    record.run1_literal_runs += block.run1_literal_runs
    record.match_after_match += block.match_after_match


def parse_stream(
    data: bytes, start_bit: int = 0, keep_tokens: bool = False, strict: bool = False,
    max_bytes: int | None = None,
) -> StreamRecord:
    """Convenience wrapper used by the feature layer (non-strict by default)."""
    return inflate(
        data,
        start_bit=start_bit,
        keep_output=True,
        keep_tokens=keep_tokens,
        strict=strict,
        max_bytes=max_bytes,
    )
