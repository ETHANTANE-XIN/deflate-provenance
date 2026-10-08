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
    sync_flush: bool = False,
) -> bytes:
    """Raw DEFLATE from CPython's zlib with every parameter exposed.

    ``sync_flush`` writes the data, then ``Z_SYNC_FLUSH`` (an empty stored
    block), then ``Z_FINISH`` (an empty fixed-code final block): the shape
    streaming writers produce when they flush before closing.
    """
    comp = zlib.compressobj(level, zlib.DEFLATED, -wbits, mem_level, strategy)
    if sync_flush:
        return comp.compress(data) + comp.flush(zlib.Z_SYNC_FLUSH) + comp.flush(zlib.Z_FINISH)
    return comp.compress(data) + comp.flush()


#: levels also produced with a sync flush before finishing (setting "<level>f"),
#: so the corpus shows that flush markers are not specific to one encoder
FLUSH_LEVELS = (1, 6, 9)


class ZlibEncoder(Encoder):
    """zlib levels 1 to 9 (proposal III.B), plus flushed variants at levels 1,
    6 and 9; strategies as optional extras."""

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
        base = [str(i) for i in range(1, 10)] + [f"{i}f" for i in FLUSH_LEVELS]
        if not self.include_strategies:
            return base
        return base + [f"6{s}" for s in ("filt", "huff", "rle", "fixed")]

    def compress(self, data: bytes, setting: str) -> EncoderResult:
        if setting.isdigit():
            n, strategy, sname = int(setting), zlib.Z_DEFAULT_STRATEGY, "def"
            raw = zlib_raw(data, n, strategy=strategy)
        elif setting.endswith("f") and setting[:-1].isdigit():
            n, sname = int(setting[:-1]), "sync-flush"
            raw = zlib_raw(data, n, sync_flush=True)
        else:
            n = int(setting[0])
            sname = setting[1:]
            raw = zlib_raw(data, n, strategy=STRATEGIES[sname])
        return self.result(raw, setting, numeric_level=n, strategy=sname)


register(ZlibEncoder())
