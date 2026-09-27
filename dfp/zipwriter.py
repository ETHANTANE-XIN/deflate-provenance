"""Lossless ZIP reader and writer for the adversarial experiments.

The producer-rewrite test, the mixed-encoder edit test and the metadata
normaliser all need to rewrite a ZIP's *metadata* while copying the
compressed bytes verbatim.  This module parses every entry through the
central directory (never by scanning for signatures, which can occur inside
compressed data), keeps each field and extra-field block, and writes a
well-formed archive back out.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field, replace

from .containers import (
    METHOD_DEFLATE,
    METHOD_STORE,
    ZIP_CENTRAL_SIG,
    ZIP_LOCAL_SIG,
    _find_eocd,
)


@dataclass
class ZipEntry:
    name: str
    method: int
    raw: bytes  #: the compressed bytes exactly as stored
    crc32: int
    uncompressed_size: int
    flags: int = 0
    version_made_by: int = 20
    version_needed: int = 20
    mod_time: int = 0
    mod_date: int = 0x0021
    extra_local: bytes = b""
    extra_central: bytes = b""
    comment: bytes = b""
    internal_attr: int = 0
    external_attr: int = 0
    extras: dict = field(default_factory=dict)

    @property
    def has_data_descriptor(self) -> bool:
        return bool(self.flags & 0x08)


def read_zip(data: bytes) -> tuple[list[ZipEntry], bytes]:
    """Parse every entry (with its compressed bytes) and the archive comment."""
    eocd = _find_eocd(data)
    (_d, _cd, count, _total, cd_size, cd_offset, comment_len) = struct.unpack_from(
        "<HHHHIIH", data, eocd + 4)
    comment = data[eocd + 22 : eocd + 22 + comment_len]
    entries: list[ZipEntry] = []
    pos = cd_offset
    for _ in range(count):
        if data[pos : pos + 4] != ZIP_CENTRAL_SIG:
            raise ValueError(f"bad central directory entry at {pos}")
        (made, need, flags, method, mtime, mdate, crc, csize, usize, nlen, xlen, clen,
         _disk, iattr, eattr, loff) = struct.unpack_from("<HHHHHHIIIHHHHHII", data, pos + 4)
        name_b = data[pos + 46 : pos + 46 + nlen]
        xc = data[pos + 46 + nlen : pos + 46 + nlen + xlen]
        ec = data[pos + 46 + nlen + xlen : pos + 46 + nlen + xlen + clen]
        pos += 46 + nlen + xlen + clen
        if data[loff : loff + 4] != ZIP_LOCAL_SIG:
            raise ValueError(f"bad local header at {loff}")
        lnlen, lxlen = struct.unpack_from("<HH", data, loff + 26)
        xl = data[loff + 30 + lnlen : loff + 30 + lnlen + lxlen]
        body = loff + 30 + lnlen + lxlen
        entries.append(ZipEntry(
            name=name_b.decode("utf-8", "replace"), method=method,
            raw=data[body : body + csize], crc32=crc, uncompressed_size=usize,
            flags=flags, version_made_by=made, version_needed=need,
            mod_time=mtime, mod_date=mdate, extra_local=xl, extra_central=xc,
            comment=ec, internal_attr=iattr, external_attr=eattr,
        ))
    return entries, comment


def write_zip(entries: list[ZipEntry], comment: bytes = b"") -> bytes:
    """Serialise entries; data descriptors are written when the flag says so."""
    out = bytearray()
    central = bytearray()
    for e in entries:
        offset = len(out)
        name_b = e.name.encode("utf-8")
        dd = e.has_data_descriptor
        out += ZIP_LOCAL_SIG
        out += struct.pack(
            "<HHHHHIIIHH", e.version_needed, e.flags, e.method, e.mod_time, e.mod_date,
            0 if dd else e.crc32, 0 if dd else len(e.raw), 0 if dd else e.uncompressed_size,
            len(name_b), len(e.extra_local))
        out += name_b + e.extra_local + e.raw
        if dd:
            out += b"PK\x07\x08" + struct.pack("<III", e.crc32, len(e.raw), e.uncompressed_size)
        central += ZIP_CENTRAL_SIG
        central += struct.pack(
            "<HHHHHHIIIHHHHHII", e.version_made_by, e.version_needed, e.flags, e.method,
            e.mod_time, e.mod_date, e.crc32, len(e.raw), e.uncompressed_size, len(name_b),
            len(e.extra_central), len(e.comment), 0, e.internal_attr, e.external_attr, offset)
        central += name_b + e.extra_central + e.comment
    cd_offset = len(out)
    out += central
    out += b"PK\x05\x06" + struct.pack("<HHHHIIH", 0, 0, len(entries), len(entries),
                                       len(central), cd_offset, len(comment))
    out += comment
    return bytes(out)


def stored_entry(name: str, data: bytes, template: ZipEntry | None = None) -> ZipEntry:
    """A method-0 (uncompressed) entry, copying header style from ``template``."""
    base = template or ZipEntry(name, METHOD_STORE, b"", 0, 0)
    return replace(base, name=name, method=METHOD_STORE, raw=data,
                   crc32=zlib.crc32(data) & 0xFFFFFFFF, uncompressed_size=len(data),
                   flags=base.flags & ~0x06)


def deflated_entry(name: str, data: bytes, raw: bytes, template: ZipEntry) -> ZipEntry:
    """A DEFLATE entry holding ``raw`` (already compressed ``data``)."""
    return replace(template, name=name, method=METHOD_DEFLATE, raw=raw,
                   crc32=zlib.crc32(data) & 0xFFFFFFFF, uncompressed_size=len(data))


def normalise_metadata(data: bytes) -> bytes:
    """Neutralise the forgeable metadata, copying compressed bytes verbatim.

    Version-made-by, the informational flag bits (the compression option bits
    1-2; the data-descriptor and UTF-8 bits are kept so the archive stays
    readable), timestamps, attributes and extra fields are all reset.
    """
    entries, _ = read_zip(data)
    out = [
        replace(e, version_made_by=20, flags=e.flags & ~0x06, mod_time=0, mod_date=0x0021,
                extra_local=b"", extra_central=b"", comment=b"", internal_attr=0,
                external_attr=0)
        for e in entries
    ]
    return write_zip(out)


def impersonate(data: bytes, template: bytes, producer_claim: str | None = None,
                claim_entry: str = "docProps/app.xml") -> bytes:
    """Give ``data`` the container metadata of ``template``'s writer.

    Every entry takes the header fields of the template's first DEFLATE entry
    (version made by, version needed, flags including the compression option
    bits and data-descriptor use, timestamps, attributes and extra-field
    blocks).  If ``producer_claim`` is given, the ``Application`` element of
    ``claim_entry`` is rewritten to it and that entry is stored uncompressed,
    so no DEFLATE stream is touched.
    """
    import re

    from .deflate import parse_stream

    entries, comment = read_zip(data)
    tmpl_entries, _ = read_zip(template)
    tmpl = next((e for e in tmpl_entries if e.method == METHOD_DEFLATE), tmpl_entries[0])
    out = []
    for e in entries:
        new = replace(e, version_made_by=tmpl.version_made_by,
                      version_needed=tmpl.version_needed,
                      flags=(tmpl.flags if e.method == METHOD_DEFLATE else tmpl.flags & ~0x06),
                      mod_time=tmpl.mod_time, mod_date=tmpl.mod_date,
                      extra_local=tmpl.extra_local, extra_central=tmpl.extra_central,
                      internal_attr=tmpl.internal_attr, external_attr=tmpl.external_attr)
        if producer_claim is not None and e.name == claim_entry:
            text = e.raw if e.method == METHOD_STORE else parse_stream(e.raw).output
            text = re.sub(rb"(<Application>)[^<]*(</Application>)",
                          lambda m: m.group(1) + producer_claim.encode() + m.group(2), text)
            new = stored_entry(e.name, text, new)
        out.append(new)
    return write_zip(out, comment)


def replace_entry(data: bytes, name: str, new_data: bytes, new_raw: bytes) -> bytes:
    """Swap one entry's content for ``new_data`` compressed as ``new_raw``."""
    entries, comment = read_zip(data)
    out = [deflated_entry(name, new_data, new_raw, e) if e.name == name else e
           for e in entries]
    return write_zip(out, comment)
