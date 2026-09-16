"""LSB-first bit reader for DEFLATE bitstreams (RFC 1951 section 3.1.1).

DEFLATE packs Huffman codes starting with the least-significant bit of each
byte, while multi-bit integer fields (LEN, extra bits, HLIT ...) are stored
LSB-first as well.  Huffman *codes* themselves are packed most-significant-bit
first, which is why ``read_bit`` is exposed separately: the Huffman decoder
consumes one bit at a time and assembles the code MSB-first itself.

The reader tracks an absolute bit position so the parser can record exact
block boundaries -- a forensic artefact in its own right, because different
encoders split blocks at different places.
"""

from __future__ import annotations


class BitStreamError(Exception):
    """Raised when the bitstream ends prematurely or is malformed."""


class BitReader:
    """Sequential LSB-first bit reader over an immutable byte buffer."""

    __slots__ = ("_data", "_len", "_pos")

    def __init__(self, data: bytes, start_bit: int = 0) -> None:
        self._data = data
        self._len = len(data) * 8
        if start_bit < 0 or start_bit > self._len:
            raise ValueError(f"start_bit {start_bit} outside buffer of {self._len} bits")
        self._pos = start_bit

    # -- position -----------------------------------------------------------
    @property
    def bit_pos(self) -> int:
        """Absolute bit offset from the start of the buffer."""
        return self._pos

    @property
    def byte_pos(self) -> int:
        """Byte offset containing the next bit."""
        return self._pos >> 3

    @property
    def bits_left(self) -> int:
        return self._len - self._pos

    @property
    def bit_in_byte(self) -> int:
        return self._pos & 7

    def seek_bit(self, pos: int) -> None:
        if pos < 0 or pos > self._len:
            raise ValueError(f"bit position {pos} outside buffer")
        self._pos = pos

    # -- reading ------------------------------------------------------------
    def read_bit(self) -> int:
        """Read a single bit, LSB of the current byte first."""
        if self._pos >= self._len:
            raise BitStreamError("end of bitstream while reading 1 bit")
        byte = self._data[self._pos >> 3]
        bit = (byte >> (self._pos & 7)) & 1
        self._pos += 1
        return bit

    def read_bits(self, count: int) -> int:
        """Read ``count`` bits as an LSB-first integer (RFC 1951 field order)."""
        if count == 0:
            return 0
        if count < 0:
            raise ValueError("count must be non-negative")
        if self._pos + count > self._len:
            raise BitStreamError(
                f"end of bitstream: wanted {count} bits, {self.bits_left} left"
            )
        value = 0
        pos = self._pos
        data = self._data
        for i in range(count):
            value |= ((data[pos >> 3] >> (pos & 7)) & 1) << i
            pos += 1
        self._pos = pos
        return value

    def align_to_byte(self) -> tuple[int, int]:
        """Discard bits up to the next byte boundary.

        Returns ``(n_bits_discarded, value_of_those_bits)``.  Encoders are free
        to emit anything here; most emit zeros, so a non-zero value is a
        fingerprint.
        """
        skew = self._pos & 7
        if skew == 0:
            return 0, 0
        n = 8 - skew
        return n, self.read_bits(n)

    def read_aligned_bytes(self, count: int) -> bytes:
        """Read whole bytes; the reader must already be byte-aligned."""
        if self._pos & 7:
            raise BitStreamError("read_aligned_bytes on a non-aligned position")
        start = self._pos >> 3
        end = start + count
        if end > len(self._data):
            raise BitStreamError(
                f"end of bitstream: wanted {count} bytes, "
                f"{len(self._data) - start} left"
            )
        self._pos = end * 8
        return self._data[start:end]

    def peek_bits(self, count: int) -> int:
        """Read ``count`` bits without advancing (short read is zero-padded)."""
        saved = self._pos
        available = min(count, self.bits_left)
        value = self.read_bits(available)
        self._pos = saved
        return value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<BitReader bit={self._pos} byte={self.byte_pos}"
            f"+{self.bit_in_byte} left={self.bits_left}>"
        )
