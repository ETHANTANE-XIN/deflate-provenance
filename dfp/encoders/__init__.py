"""Encoder adapters.

Each adapter compresses arbitrary bytes to a *raw* DEFLATE stream (no zlib or
gzip wrapper) and reports a stable label ``family/level`` plus a probe of
whether the backend is actually available on this machine.

An adapter never appears in the corpus unless :func:`available` returns True, so
the corpus is honest about what was really used to build it.

Families implemented
---------------------
``zlib``       CPython's zlib, ten levels x five strategies (the RFC 1951 anchor)
``purepy``     our own pure-Python encoder -- greedy and lazy variants; doubles
               as the held-out "unknown" for the open-set test and as the host
               for the covert channel
``java``       Java 17 ``java.util.zip.Deflater`` (a different implementation)
``dotnet``     .NET ``System.IO.Compression.DeflateStream``
``node``       Node.js ``zlib.deflateRawSync`` (zlib, but a distinct build/version)
``libarchive`` bsdtar / libarchive's DEFLATE
"""

from __future__ import annotations

from .base import Encoder, EncoderResult, list_encoders, get_encoder

__all__ = ["Encoder", "EncoderResult", "list_encoders", "get_encoder"]
