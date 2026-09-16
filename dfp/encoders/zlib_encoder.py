"""CPython zlib adapter -- the RFC 1951 anchor family."""

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


class ZlibEncoder(Encoder):
    family = "zlib"

    def __init__(self, include_strategies: bool = True) -> None:
        self.include_strategies = include_strategies

    def available(self) -> bool:
        return True

    def levels(self) -> list[str]:
        base = [str(i) for i in range(0, 10)]
        if not self.include_strategies:
            return base
        # keep the corpus tractable: default strategy over all levels, plus the
        # non-default strategies at a representative level 6
        extra = [f"6{s}" for s in ("filt", "huff", "rle", "fixed")]
        return base + extra

    def compress(self, data: bytes, level: str) -> EncoderResult:
        if level.isdigit():
            n, strategy, sname = int(level), zlib.Z_DEFAULT_STRATEGY, "def"
        else:
            n = int(level[0])
            sname = level[1:]
            strategy = STRATEGIES[sname]
        comp = zlib.compressobj(n, zlib.DEFLATED, -15, 8, strategy)
        raw = comp.compress(data) + comp.flush()
        return EncoderResult(
            raw_deflate=raw,
            family=self.family,
            level=level,
            label=self.label(level),
            extra={
                "runtime_version": zlib.ZLIB_RUNTIME_VERSION,
                "numeric_level": n,
                "strategy": sname,
            },
        )


register(ZlibEncoder())
