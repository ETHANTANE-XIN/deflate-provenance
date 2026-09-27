"""Encoder adapters.

Each adapter compresses arbitrary bytes to a *raw* DEFLATE stream (no zlib or
gzip wrapper), reports the exact version that produced it, and declares the
DEFLATE library its program is built on (see :mod:`dfp.encoders.base`).

Adapters (proposal section III.B)
---------------------------------
``zlib``        CPython's zlib, levels 1 to 9 -- reference of the zlib profile
``java``        ``java.util.zip.Deflater``, levels 1 to 9 -- built on zlib
``zlib-ng``     zlib-ng via python-zlib-ng
``node``        Node.js, which bundles Chromium's zlib fork
``libdeflate``  libdeflate via python-deflate
``zopfli``      Google's zopfli via python-zopfli
``7zip``        7-Zip's own DEFLATE encoder
``go``          Go's ``compress/flate``
``dotnet``      .NET ``DeflateStream`` (zlib-ng from .NET 9)
``libarchive``  bsdtar's ZIP writer (links the system zlib)
``purepy``      the team's own pure-Python encoder: never a training profile,
                used as an extra unseen encoder and by the covert-channel
                extension

Applications whose encoder cannot be called as a library, such as Microsoft
Word, are profiled from saved documents instead (see :mod:`dfp.realfiles`).
"""

from __future__ import annotations

from .base import (
    Encoder,
    EncoderResult,
    get_encoder,
    list_encoders,
    reference_for,
)

__all__ = ["Encoder", "EncoderResult", "list_encoders", "get_encoder", "reference_for"]
