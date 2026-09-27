"""In-process adapters for zlib-ng, libdeflate and zopfli.

Each uses a Python binding of the real C library (see ``requirements.txt``),
so the corpus contains the genuine encoders rather than re-implementations.
An adapter whose binding is not installed reports itself unavailable and
never enters the corpus.
"""

from __future__ import annotations

import importlib.metadata

from .base import Encoder, EncoderResult, register


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


class ZlibNgEncoder(Encoder):
    """zlib-ng (the library .NET 9 and later also use)."""

    name = "zlib-ng"
    library = "zlib-ng"
    reference = True

    def _mod(self):
        try:
            from zlib_ng import zlib_ng
        except ImportError:
            return None
        return zlib_ng

    def available(self) -> bool:
        return self._mod() is not None

    def version(self) -> str:
        mod = self._mod()
        if mod is None:
            return "not installed"
        return f"zlib-ng {mod.ZLIBNG_RUNTIME_VERSION} (python-zlib-ng {_pkg_version('zlib-ng')})"

    def settings(self) -> list[str]:
        return ["1", "3", "6", "9"]

    def compress(self, data: bytes, setting: str) -> EncoderResult:
        mod = self._mod()
        comp = mod.compressobj(int(setting), mod.DEFLATED, -15, 8, mod.Z_DEFAULT_STRATEGY)
        return self.result(comp.compress(data) + comp.flush(), setting)


class LibdeflateEncoder(Encoder):
    """libdeflate, levels 1 to 12 (a representative subset is used)."""

    name = "libdeflate"
    library = "libdeflate"
    reference = True

    def _mod(self):
        try:
            import deflate
        except ImportError:
            return None
        return deflate

    def available(self) -> bool:
        return self._mod() is not None

    def version(self) -> str:
        if self._mod() is None:
            return "not installed"
        return f"libdeflate via python-deflate {_pkg_version('deflate')}"

    def settings(self) -> list[str]:
        return ["1", "6", "9", "12"]

    def compress(self, data: bytes, setting: str) -> EncoderResult:
        raw = self._mod().deflate_compress(data, int(setting))
        return self.result(raw, setting)


class ZopfliEncoder(Encoder):
    """Google's zopfli (exhaustive optimisation; slow but distinctive)."""

    name = "zopfli"
    library = "zopfli"
    reference = True

    def _mod(self):
        try:
            from zopfli import zopfli
        except ImportError:
            return None
        return zopfli

    def available(self) -> bool:
        return self._mod() is not None

    def version(self) -> str:
        if self._mod() is None:
            return "not installed"
        return f"zopfli via python-zopfli {_pkg_version('zopfli')}"

    def settings(self) -> list[str]:
        # numiterations; 15 is zopfli's default
        return ["i5", "i15"]

    def compress(self, data: bytes, setting: str) -> EncoderResult:
        wrapped = self._mod().compress(data, numiterations=int(setting[1:]), gzip_mode=0)
        # zlib container: strip the 2-byte header and the 4-byte Adler-32
        return self.result(bytes(wrapped[2:-4]), setting)


register(ZlibNgEncoder())
register(LibdeflateEncoder())
register(ZopfliEncoder())
