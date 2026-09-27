"""CPython zlib adapter -- the reference implementation of the zlib profile."""

from __future__ import annotations

import zlib

from .base import Encoder, EncoderResult, register

STRATEGIES = {
    "def": zlib.Z_DEFAULT_STRATEGY,
    "filt": zlib.Z_FILTERED,
    "huff": zlib.Z_HUFFMAN_ONLY,
    "rle": zlib.Z_RLE,
    "fixed": zlib.Z_FIXED,
}


def zlib_raw(
    data: bytes,
    level: int,
    mem_level: int = 8,
    wbits: int = 15,
    strategy: int = zlib.Z_DEFAULT_STRATEGY,
) -> bytes:
    """Raw DEFLATE from CPython's zlib with every parameter exposed."""
    comp = zlib.compressobj(level, zlib.DEFLATED, -wbits, mem_level, strategy)
    return comp.compress(data) + comp.flush()


class ZlibEncoder(Encoder):
    """zlib levels 1 to 9 (proposal III.B); strategies as optional extras."""

    name = "zlib"
    library = "zlib"
    reference = True

    def __init__(self, include_strategies: bool = False) -> None:
        self.include_strategies = include_strategies

    def available(self) -> bool:
        return True

    def version(self) -> str:
        return f"zlib {zlib.ZLIB_RUNTIME_VERSION}"

    def settings(self) -> list[str]:
        base = [str(i) for i in range(1, 10)]
        if not self.include_strategies:
            return base
        return base + [f"6{s}" for s in ("filt", "huff", "rle", "fixed")]

    def compress(self, data: bytes, setting: str) -> EncoderResult:
        if setting.isdigit():
            n, strategy, sname = int(setting), zlib.Z_DEFAULT_STRATEGY, "def"
        else:
            n = int(setting[0])
            sname = setting[1:]
            strategy = STRATEGIES[sname]
        raw = zlib_raw(data, n, strategy=strategy)
        return self.result(raw, setting, numeric_level=n, strategy=sname)


register(ZlibEncoder())
