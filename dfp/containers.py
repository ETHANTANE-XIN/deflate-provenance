"""Locate raw DEFLATE streams inside real-world containers.

Supported: ZIP and every ZIP-based format (DOCX / XLSX / PPTX / APK / JAR /
EPUB / ODF), GZIP, zlib-wrapped streams, PNG (concatenated IDAT), and headerless
raw DEFLATE.

The extractor deliberately also harvests the *container metadata* that
traditional provenance work relies on -- version-made-by, DOS timestamps, extra
field IDs, entry order, GZIP OS/XFL bytes.  The point of the project is to
compare that easily-forged evidence against the bitstream evidence, so both
must be captured side by side.

Everything is parsed by hand from the format specifications; ``zipfile`` is used
only where we need a decompressed *entry* for a recompression experiment, never
to obtain the compressed bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

ZIP_LOCAL_SIG = b"PK\x03\x04"
ZIP_CENTRAL_SIG = b"PK\x01\x02"
ZIP_EOCD_SIG = b"PK\x05\x06"
ZIP64_EOCD_SIG = b"PK\x06\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
GZIP_MAGIC = b"\x1f\x8b"
PDF_MAGIC = b"%PDF-"

METHOD_STORE = 0
METHOD_DEFLATE = 8

#: ZIP "version made by" upper byte -> host system (APPNOTE 4.4.2)
ZIP_HOST_SYSTEM = {
    0: "MS-DOS/FAT",
    1: "Amiga",
    3: "Unix",
    7: "Macintosh",
    10: "NTFS",
    11: "MVS",
    14: "VFAT",
    19: "OS X (Darwin)",
}

#: Common ZIP extra-field header IDs, a classic metadata provenance signal
ZIP_EXTRA_NAMES = {
    0x0001: "Zip64",
    0x000A: "NTFS-times",
    0x5455: "extended-timestamp(UT)",
    0x7875: "Unix-uid-gid(ux)",
    0x9901: "AES-x",
    0xCAFE: "Java-executable",
    0x5855: "Info-ZIP-Unix-old",
    0x6375: "Info-ZIP-Unicode-comment",
    0x7075: "Info-ZIP-Unicode-path",
}

#: GZIP OS byte (RFC 1952 section 2.3.1)
GZIP_OS = {
    0: "FAT",
    3: "Unix",
    7: "Macintosh",
    11: "NTFS",
    13: "Acorn RISCOS",
    255: "unknown",
}

CONTAINER_BY_SUFFIX = {
    ".zip": "zip",
    ".docx": "ooxml",
    ".xlsx": "ooxml",
    ".pptx": "ooxml",
    ".docm": "ooxml",
    ".apk": "apk",
    ".jar": "jar",
    ".aar": "jar",
    ".epub": "epub",
    ".odt": "odf",
    ".ods": "odf",
    ".odp": "odf",
    ".gz": "gzip",
    ".tgz": "gzip",
    ".png": "png",
    ".pdf": "pdf",
    ".zz": "zlib",
    ".deflate": "raw",
    ".raw": "raw",
}


class ContainerError(Exception):
    """Raised when a container cannot be parsed."""


@dataclass
class DeflateStream:
    """One DEFLATE stream located inside a container."""

    payload: bytes  #: buffer containing the stream
    start_bit: int = 0  #: bit offset of the first block header within payload
    name: str = ""  #: entry name / stream label
    container: str = "raw"  #: container kind
    source: str = ""  #: file the stream came from
    index: int = 0  #: ordinal within the container
    declared_crc32: int | None = None
    declared_usize: int | None = None
    declared_csize: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.payload) - (self.start_bit // 8)

    def label(self) -> str:
        return f"{self.container}:{self.name or self.index}"


@dataclass
class ContainerReport:
    """Streams plus container-level metadata for one input file."""

    path: str
    container: str
    streams: list[DeflateStream] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    stored_entries: int = 0
    other_method_entries: int = 0
    #: entries whose data is encrypted (general-purpose flag bit 0 or 6): their
    #: bytes are ciphertext, not DEFLATE, so no stream is built for them
    encrypted_entries: int = 0
    notes: list[str] = field(default_factory=list)
    #: every archive entry, whatever its method (name, method, sizes, flags)
    entries: list[dict] = field(default_factory=list)
    #: producer claims found in the container (see :func:`harvest_claims`)
    claims: dict = field(default_factory=dict)


# --- helpers ---------------------------------------------------------------


def _dos_datetime(dt: int, dd: int) -> str:
    second = (dt & 0x1F) * 2
    minute = (dt >> 5) & 0x3F
    hour = (dt >> 11) & 0x1F
    day = dd & 0x1F
    month = (dd >> 5) & 0x0F
    year = ((dd >> 9) & 0x7F) + 1980
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def _parse_extra(blob: bytes) -> list[dict]:
    fields: list[dict] = []
    pos = 0
    while pos + 4 <= len(blob):
        header_id, size = struct.unpack_from("<HH", blob, pos)
        pos += 4
        fields.append(
            {
                "id": f"0x{header_id:04X}",
                "name": ZIP_EXTRA_NAMES.get(header_id, "unknown"),
                "size": size,
            }
        )
        pos += size
    return fields


def _zip_kind(path: str | Path | None) -> str:
    if path:
        kind = CONTAINER_BY_SUFFIX.get(Path(path).suffix.lower())
        if kind and kind not in ("gzip", "png", "zlib", "raw", "pdf"):
            return kind
    return "zip"


def detect_container(data: bytes, path: str | Path | None = None) -> str:
    """Identify the container from magic bytes, falling back to the suffix.

    A ZIP whose first bytes are something else (a self-extracting stub, or data
    prepended to the archive) is recognised from its end-of-central-directory
    record, as ZIP readers do.
    """
    if data[:4] in (ZIP_LOCAL_SIG, ZIP_CENTRAL_SIG, ZIP_EOCD_SIG):
        return _zip_kind(path)
    if data[:2] == GZIP_MAGIC:
        return "gzip"
    if data[:8] == PNG_MAGIC:
        return "png"
    if data[:5] == PDF_MAGIC:
        return "pdf"
    if _looks_like_zip(data):
        return _zip_kind(path)
    if len(data) >= 2 and data[0] & 0x0F == 8 and ((data[0] << 8) | data[1]) % 31 == 0:
        return "zlib"
    if path:
        return CONTAINER_BY_SUFFIX.get(Path(path).suffix.lower(), "raw")
    return "raw"


def _looks_like_zip(data: bytes) -> bool:
    """True when an EOCD record near the end points at a central directory."""
    try:
        eocd = _find_eocd(data)
    except ContainerError:
        return False
    if eocd + 22 > len(data):
        return False
    cd_size, cd_offset = struct.unpack_from("<II", data, eocd + 12)
    z64 = _zip64_eocd(data, eocd)
    end_of_cd = eocd
    if z64 is not None:
        cd_size, cd_offset, end_of_cd = z64["cd_size"], z64["cd_offset"], z64["record_offset"]
    if data[cd_offset : cd_offset + 4] == ZIP_CENTRAL_SIG:
        return True
    cand = end_of_cd - cd_size
    return 0 <= cand < len(data) and data[cand : cand + 4] == ZIP_CENTRAL_SIG


# --- ZIP -------------------------------------------------------------------


def _find_eocd(data: bytes) -> int:
    """Locate the end-of-central-directory record, allowing for a comment."""
    limit = min(len(data), 65535 + 22)
    window = data[-limit:]
    idx = window.rfind(ZIP_EOCD_SIG)
    if idx < 0:
        raise ContainerError("no ZIP end-of-central-directory record found")
    return len(data) - limit + idx


def _zip64_eocd(data: bytes, eocd: int) -> dict | None:
    """Read the ZIP64 end-of-central-directory record (APPNOTE 4.3.14-4.3.15).

    The ZIP64 locator sits immediately before the classic EOCD record and points
    at the ZIP64 record.  When data was prepended without adjusting offsets the
    stored offset is wrong, so the record is then looked for just before the
    locator, where every writer puts it.
    """
    loc = eocd - 20
    if loc < 0 or data[loc : loc + 4] != ZIP64_LOCATOR_SIG:
        return None
    _disk, stored, _ndisks = struct.unpack_from("<IQI", data, loc + 4)
    pos = stored if data[stored : stored + 4] == ZIP64_EOCD_SIG else data.rfind(ZIP64_EOCD_SIG, 0, loc)
    if pos < 0 or pos + 56 > len(data):
        return None
    (
        _size, _made, _need, _disk_no, _cd_disk,
        _entries_disk, entries, cd_size, cd_offset,
    ) = struct.unpack_from("<QHHIIQQQQ", data, pos + 4)
    return {
        "record_offset": pos,
        "stored_offset": stored,
        "entries": entries,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
    }


def _zip64_sizes(extra: bytes, usize: int, csize: int, local_offset: int) -> tuple[int, int, int]:
    """Resolve 0xFFFFFFFF fields from a central entry's ZIP64 extra field (0x0001).

    The field holds, in this order, only the values whose 32-bit slot is
    saturated: uncompressed size, compressed size, local header offset.
    """
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4 : pos + 4 + size]
        pos += 4 + size
        if header_id != 0x0001:
            continue
        values = [struct.unpack_from("<Q", body, i)[0] for i in range(0, len(body) - 7, 8)]
        if usize == 0xFFFFFFFF and values:
            usize = values.pop(0)
        if csize == 0xFFFFFFFF and values:
            csize = values.pop(0)
        if local_offset == 0xFFFFFFFF and values:
            local_offset = values.pop(0)
        break
    return usize, csize, local_offset


def extract_zip(data: bytes, path: str, container: str) -> ContainerReport:
    report = ContainerReport(path=path, container=container)
    eocd = _find_eocd(data)
    (
        _disk,
        _cd_disk,
        entries_here,
        _entries_total,
        cd_size,
        cd_offset,
        comment_len,
    ) = struct.unpack_from("<HHHHIIH", data, eocd + 4)
    comment = data[eocd + 22 : eocd + 22 + comment_len]

    # ZIP64: more than 65535 entries or offsets/sizes beyond 4 GiB
    z64 = _zip64_eocd(data, eocd)
    end_of_cd = eocd
    if z64 is not None:
        entries_here = z64["entries"]
        cd_size = z64["cd_size"]
        cd_offset = z64["cd_offset"]
        end_of_cd = z64["record_offset"]

    # Data in front of the archive whose offsets were not adjusted (a
    # self-extracting stub added with `cat`, for example) shifts everything by
    # the same amount; ZIP readers find the central directory from the end.
    shift = 0
    if entries_here and data[cd_offset : cd_offset + 4] != ZIP_CENTRAL_SIG:
        cand = end_of_cd - cd_size
        if 0 <= cand < len(data) and data[cand : cand + 4] == ZIP_CENTRAL_SIG:
            shift = cand - cd_offset
            cd_offset = cand
            report.notes.append(
                f"{shift} bytes precede the ZIP structure and its offsets were not "
                "adjusted (self-extracting stub or prepended data); offsets were rebased"
            )

    report.metadata.update(
        {
            "eocd_offset": eocd,
            "central_directory_offset": cd_offset,
            "central_directory_size": cd_size,
            "entry_count": entries_here,
            "archive_comment_len": comment_len,
            "archive_comment": comment[:200].decode("utf-8", "replace"),
            "zip64": z64 is not None,
            "offset_shift": shift,
        }
    )

    pos = cd_offset
    order: list[str] = []
    host_systems: set[str] = set()
    versions: set[int] = set()
    extra_ids: set[str] = set()
    flags_seen: set[int] = set()
    timestamps: list[str] = []
    name_counts: dict[str, int] = {}
    backslash_names = 0
    index = 0
    for _ in range(entries_here):
        if data[pos : pos + 4] != ZIP_CENTRAL_SIG:
            report.notes.append(f"central directory truncated at {pos}")
            break
        (
            version_made,
            _version_need,
            flags,
            method,
            mtime,
            mdate,
            crc,
            csize,
            usize,
            name_len,
            extra_len,
            comment_len_e,
            _disk_start,
            internal_attr,
            external_attr,
            local_offset,
        ) = struct.unpack_from("<HHHHHHIIIHHHHHII", data, pos + 4)
        name = data[pos + 46 : pos + 46 + name_len].decode("utf-8", "replace")
        extra = data[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        pos += 46 + name_len + extra_len + comment_len_e
        usize, csize, local_offset = _zip64_sizes(extra, usize, csize, local_offset)
        local_offset += shift

        # APPNOTE 4.4.17 requires '/' separators; PowerShell 5.1's
        # Compress-Archive writes '\'.  Lookups use the normalised name.
        normalised = name.replace("\\", "/")
        if normalised != name:
            backslash_names += 1
        name_counts[normalised] = name_counts.get(normalised, 0) + 1

        order.append(name)
        encrypted = bool(flags & 0x01) or bool(flags & 0x40) or method == 99
        entry = {
            "name": name,
            "name_normalised": normalised,
            "method": method,
            "compressed_size": csize,
            "uncompressed_size": usize,
            "flags": flags,
            "option_bits": (flags >> 1) & 0x3,
            "version_made_by": version_made,
            "local_header_offset": local_offset,
            "crc32": crc,
            "encrypted": encrypted,
            "duplicate_of": None,
            "stream_index": None,
        }
        report.entries.append(entry)
        host = version_made >> 8
        host_systems.add(ZIP_HOST_SYSTEM.get(host, f"host{host}"))
        versions.add(version_made & 0xFF)
        flags_seen.add(flags)
        parsed_extra = _parse_extra(extra)
        extra_ids.update(f["name"] for f in parsed_extra)
        timestamps.append(_dos_datetime(mtime, mdate))

        if encrypted:
            # the bytes are ciphertext: never parse or classify them
            report.encrypted_entries += 1
            continue
        if method != METHOD_DEFLATE:
            if method == METHOD_STORE:
                report.stored_entries += 1
            else:
                report.other_method_entries += 1
            continue

        # Compressed bytes start after the *local* header, whose name/extra
        # lengths may differ from the central directory's.
        if data[local_offset : local_offset + 4] != ZIP_LOCAL_SIG:
            report.notes.append(f"bad local header for {name!r} at {local_offset}")
            continue
        l_name_len, l_extra_len = struct.unpack_from("<HH", data, local_offset + 26)
        body = local_offset + 30 + l_name_len + l_extra_len
        if csize == 0xFFFFFFFF:
            # saturated size with no ZIP64 field: bound it by the next header
            end = data.find(ZIP_LOCAL_SIG, body)
            if end < 0:
                end = cd_offset
            payload = data[body:end]
        else:
            # the central directory's size is authoritative, including when the
            # local header defers it to a data descriptor
            payload = data[body : body + csize]

        entry["stream_index"] = len(report.streams)
        report.streams.append(
            DeflateStream(
                payload=payload,
                name=name,
                container=container,
                source=path,
                index=index,
                declared_crc32=crc,
                declared_usize=usize,
                declared_csize=csize,
                metadata={
                    "version_made_by": version_made & 0xFF,
                    "host_system": ZIP_HOST_SYSTEM.get(host, f"host{host}"),
                    "general_purpose_flags": flags,
                    "compression_level_hint_bits": (flags >> 1) & 0x3,
                    "has_data_descriptor": bool(flags & 0x08),
                    "utf8_names": bool(flags & 0x800),
                    "dos_time": _dos_datetime(mtime, mdate),
                    "internal_attr": internal_attr,
                    "external_attr": external_attr,
                    "extra_fields": parsed_extra,
                    "local_header_offset": local_offset,
                },
            )
        )
        index += 1

    # Two entries with one name: legitimate writers never produce this (Python's
    # zipfile does when a file is appended to edit a part), and readers disagree
    # about which copy they show.  Mark every later copy.
    duplicates = sorted(n for n, c in name_counts.items() if c > 1)
    first_index: dict[str, int] = {}
    for i, e in enumerate(report.entries):
        key = e["name_normalised"]
        if key in first_index:
            e["duplicate_of"] = first_index[key]
        else:
            first_index[key] = i
    if duplicates:
        report.notes.append(
            f"{len(duplicates)} entry name(s) occur more than once: "
            + ", ".join(duplicates[:5]) + (" ..." if len(duplicates) > 5 else "")
        )
    if backslash_names:
        report.notes.append(
            f"{backslash_names} entry name(s) use '\\' separators, which APPNOTE "
            "forbids (Windows PowerShell 5.1 Compress-Archive writes them)"
        )
    local_offsets = [e["local_header_offset"] for e in report.entries]
    report.metadata.update(
        {
            "entry_order": order[:200],
            "host_systems": sorted(host_systems),
            "version_made_by": sorted(versions),
            "general_purpose_flags": sorted(flags_seen),
            "extra_field_kinds": sorted(extra_ids),
            "first_timestamp": timestamps[0] if timestamps else None,
            "distinct_timestamps": len(set(timestamps)),
            "stored_entries": report.stored_entries,
            "encrypted_entries": report.encrypted_entries,
            "deflate_entries": len(report.streams),
            "option_bits_seen": sorted({e["option_bits"] for e in report.entries
                                        if e["method"] == METHOD_DEFLATE}),
            "duplicate_names": duplicates[:50],
            "backslash_names": backslash_names,
            # bytes before the first entry (a stub, or data prepended to the archive)
            "prepended_bytes": min(local_offsets) if local_offsets else 0,
        }
    )
    report.claims = harvest_claims(data, report)
    return report


# --- producer claims ----------------------------------------------------------

#: option bits 1-2 of the general-purpose flag for DEFLATE (APPNOTE 4.4.4)
OPTION_BITS = {0: "normal", 1: "maximum", 2: "fast", 3: "super fast"}


def read_entry(data: bytes, entry: dict, limit: int = 4_000_000) -> bytes | None:
    """Return an entry's uncompressed bytes (stored or DEFLATE) or ``None``.

    DEFLATE entries are decoded with this project's own parser, never zlib.
    """
    from .deflate import parse_stream

    if entry.get("encrypted"):
        return None
    off = entry["local_header_offset"]
    if data[off : off + 4] != ZIP_LOCAL_SIG:
        return None
    l_name_len, l_extra_len = struct.unpack_from("<HH", data, off + 26)
    body = off + 30 + l_name_len + l_extra_len
    size = entry["compressed_size"]
    if size in (0, 0xFFFFFFFF) or size > limit:
        return None
    blob = data[body : body + size]
    if entry["method"] == METHOD_STORE:
        return blob
    if entry["method"] == METHOD_DEFLATE:
        rec = parse_stream(blob, strict=False)
        return rec.output if rec.error is None else None
    return None


def _xml_text(blob: bytes, tag: str) -> str | None:
    import re

    m = re.search(rb"<(?:\w+:)?" + tag.encode() + rb"[^>]*>([^<]{0,300})</", blob)
    return m.group(1).decode("utf-8", "replace").strip() if m else None


def harvest_claims(data: bytes, report: ContainerReport) -> dict:
    """Collect what the container says about the program that wrote it.

    * OOXML (DOCX/XLSX/PPTX): ``docProps/app.xml`` ``Application`` and
      ``AppVersion``;
    * ODF: ``meta.xml`` ``meta:generator``;
    * JAR/APK: ``META-INF/MANIFEST.MF`` ``Created-By``.

    These are the claims the consistency checks compare with the compressed
    data; they are as easy to forge as any other metadata, which is the point.

    Names are matched case-insensitively with '\\' read as '/' (OPC part names
    are case-insensitive, and Compress-Archive writes backslashes).  When a
    name occurs more than once the *first* copy in central-directory order is
    used and the duplication is recorded, so the choice is explicit.
    """
    by_name: dict[str, dict] = {}
    for e in report.entries:
        by_name.setdefault(e.get("name_normalised", e["name"]).lower(), e)
    claims: dict = {}
    app = by_name.get("docprops/app.xml")
    if app:
        blob = read_entry(data, app)
        if blob:
            claims["producer"] = _xml_text(blob, "Application")
            claims["producer_version"] = _xml_text(blob, "AppVersion")
            claims["source"] = "docProps/app.xml"
    meta = by_name.get("meta.xml")
    if meta and "producer" not in claims:
        blob = read_entry(data, meta)
        if blob:
            claims["producer"] = _xml_text(blob, "generator")
            claims["source"] = "meta.xml"
    manifest = by_name.get("meta-inf/manifest.mf")
    if manifest and "producer" not in claims:
        blob = read_entry(data, manifest)
        if blob:
            for line in _manifest_lines(blob):
                if line.lower().startswith("created-by:"):
                    claims["producer"] = line.split(":", 1)[1].strip()
                    claims["source"] = "META-INF/MANIFEST.MF"
                    break
    if claims.get("producer") is None:
        claims.pop("producer", None)
    if claims and claims.get("source"):
        source_key = claims["source"].lower()
        copies = sum(1 for e in report.entries
                     if e.get("name_normalised", e["name"]).lower() == source_key)
        if copies > 1:
            claims["source_copies"] = copies
    return claims


def _manifest_lines(blob: bytes) -> list[str]:
    """Header lines of a JAR manifest with 72-byte continuations joined.

    The JAR specification wraps long values: a line starting with a single
    space continues the previous one.
    """
    lines: list[str] = []
    for raw in blob.decode("utf-8", "replace").splitlines():
        if raw.startswith(" ") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


# --- GZIP / zlib / PNG / raw ----------------------------------------------


def _gzip_member_header(data: bytes, start: int) -> tuple[dict, int]:
    """Parse one RFC 1952 member header; return its metadata and payload start."""
    if data[start : start + 2] != GZIP_MAGIC:
        raise ContainerError(f"no GZIP member header at {start}")
    method = data[start + 2]
    if method != METHOD_DEFLATE:
        raise ContainerError(f"GZIP compression method {method} is not DEFLATE")
    flags = data[start + 3]
    mtime = int.from_bytes(data[start + 4 : start + 8], "little")
    xfl = data[start + 8]
    os_byte = data[start + 9]
    pos = start + 10
    extra_meta: dict = {}
    try:
        if flags & 0x04:  # FEXTRA
            xlen = int.from_bytes(data[pos : pos + 2], "little")
            extra_meta["fextra_len"] = xlen
            pos += 2 + xlen
        name = None
        if flags & 0x08:  # FNAME
            end = data.index(b"\x00", pos)
            name = data[pos:end].decode("latin-1")
            pos = end + 1
        if flags & 0x10:  # FCOMMENT
            end = data.index(b"\x00", pos)
            extra_meta["fcomment"] = data[pos:end].decode("latin-1", "replace")
            pos = end + 1
    except ValueError as exc:
        raise ContainerError("truncated GZIP header") from exc
    if flags & 0x02:  # FHCRC
        pos += 2
    meta = {
        "flags": flags,
        "mtime": mtime,
        "xfl": xfl,
        "xfl_meaning": {2: "best-compression", 4: "fastest"}.get(xfl, "unspecified"),
        "os_byte": os_byte,
        "os": GZIP_OS.get(os_byte, f"os{os_byte}"),
        "embedded_name": name,
        "header_offset": start,
        "header_len": pos - start,
        **extra_meta,
    }
    return meta, pos


def extract_gzip(data: bytes, path: str) -> ContainerReport:
    """Extract every member of a GZIP file.

    RFC 1952 allows several members back to back (``cat a.gz b.gz``, appended
    logs); each has its own header, DEFLATE stream and CRC32/ISIZE trailer, and
    each may come from a different encoder, so every member becomes its own
    stream.  The end of each member's DEFLATE data is found with this project's
    parser; bytes after the last member are reported.
    """
    from .deflate import inflate

    report = ContainerReport(path=path, container="gzip")
    if data[:2] != GZIP_MAGIC:
        raise ContainerError("not a GZIP file")
    base = Path(path).stem
    start = 0
    member = 0
    while True:
        meta, body = _gzip_member_header(data, start)
        rec = inflate(data, start_bit=body * 8, keep_output=False, strict=False)
        name = meta["embedded_name"] or base
        label = name if member == 0 else f"{name}#{member + 1}"
        if rec.error is not None:
            # damaged or truncated member: hand its bytes to the analyser,
            # which reports the error, and stop looking for further members
            stop = len(data) - 8 if member == 0 and len(data) - 8 > body else len(data)
            report.streams.append(
                DeflateStream(payload=data[body:stop], name=label, container="gzip",
                              source=path, index=member, metadata=meta)
            )
            report.notes.append(f"GZIP member {member + 1} could not be parsed ({rec.error})")
            member += 1
            break
        end = rec.end_bit // 8  # first byte after the final block's padding
        trailer = data[end : end + 8]
        crc = int.from_bytes(trailer[:4], "little") if len(trailer) == 8 else None
        isize = int.from_bytes(trailer[4:], "little") if len(trailer) == 8 else None
        report.streams.append(
            DeflateStream(
                payload=data[body:end],
                name=label,
                container="gzip",
                source=path,
                index=member,
                declared_crc32=crc,
                declared_usize=isize,
                declared_csize=end - body,
                metadata=meta,
            )
        )
        member += 1
        start = end + 8
        if start >= len(data):
            break
        if data[start : start + 2] != GZIP_MAGIC:
            rest = data[start:]
            report.metadata["trailing_bytes"] = len(rest)
            report.notes.append(
                f"{len(rest)} bytes follow the last GZIP member"
                + (" (all zero padding)" if not rest.strip(b"\x00") else "")
            )
            break

    first = report.streams[0].metadata if report.streams else {}
    report.metadata = {**first, **report.metadata, "members": member}
    if member > 1:
        report.notes.append(f"the file holds {member} GZIP members, analysed separately")
    return report


def extract_zlib(data: bytes, path: str) -> ContainerReport:
    report = ContainerReport(path=path, container="zlib")
    cmf, flg = data[0], data[1]
    cinfo = cmf >> 4
    flevel = flg >> 6
    metadata = {
        "cmf": cmf,
        "flg": flg,
        "window_bits": cinfo + 8,
        "flevel": flevel,
        "flevel_meaning": {
            0: "fastest",
            1: "fast",
            2: "default",
            3: "maximum/slowest",
        }[flevel],
        "fdict": bool(flg & 0x20),
    }
    report.metadata = dict(metadata)
    report.streams.append(
        DeflateStream(
            payload=data[2 : len(data) - 4],
            name=Path(path).stem,
            container="zlib",
            source=path,
            declared_crc32=None,
            metadata=metadata,
        )
    )
    return report


def extract_png(data: bytes, path: str) -> ContainerReport:
    report = ContainerReport(path=path, container="png")
    pos = 8
    idat = bytearray()
    chunks: list[str] = []
    header: dict = {}
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        ctype = data[pos + 4 : pos + 8].decode("latin-1")
        body = data[pos + 8 : pos + 8 + length]
        chunks.append(ctype)
        if ctype == "IHDR" and length >= 13:
            width, height = struct.unpack_from(">II", body, 0)
            header = {
                "width": width,
                "height": height,
                "bit_depth": body[8],
                "colour_type": body[9],
                "compression": body[10],
                "filter": body[11],
                "interlace": body[12],
            }
        elif ctype == "IDAT":
            idat += body
        elif ctype == "IEND":
            break
        pos += 12 + length
    if not idat:
        raise ContainerError("PNG has no IDAT data")
    report.metadata = {"chunks": chunks, **header}
    report.streams.append(
        DeflateStream(
            payload=bytes(idat[2:]),  # skip the zlib CMF/FLG header
            name="IDAT",
            container="png",
            source=path,
            metadata={"chunk_count": chunks.count("IDAT"), **header},
        )
    )
    return report


def extract_pdf(data: bytes, path: str) -> ContainerReport:
    """Extract every ``/FlateDecode`` stream from a PDF.

    PDFs carry page content, fonts, images and cross-reference tables in
    ``stream ... endstream`` objects, most of them DEFLATE-compressed via the
    ``/FlateDecode`` filter -- which makes a PDF one of the richest DEFLATE
    provenance targets there is (a document rewritten by a different producer
    shows it in these streams).

    This is a deliberately tolerant scanner rather than a full PDF parser: it
    walks ``stream``/``endstream`` pairs and keeps the ones whose preceding
    dictionary names ``/FlateDecode``.  ``/Length`` is used when it is a direct
    integer, otherwise the ``endstream`` keyword bounds the data.  Streams that
    chain another filter before Flate (``/Filter [/ASCII85Decode /FlateDecode]``)
    are skipped, because the bytes on disk are then not raw DEFLATE.
    """
    import re

    report = ContainerReport(path=path, container="pdf")
    header = data[:16].split(b"\n", 1)[0].decode("latin-1", "replace").strip()
    producer = None
    m = re.search(rb"/Producer\s*\(([^)]{0,200})\)", data)
    if m:
        producer = m.group(1).decode("latin-1", "replace")
    creator = None
    m = re.search(rb"/Creator\s*\(([^)]{0,200})\)", data)
    if m:
        creator = m.group(1).decode("latin-1", "replace")

    report.metadata = {
        "pdf_version": header,
        "producer": producer,
        "creator": creator,
        "linearized": b"/Linearized" in data[:4096],
        "encrypted": b"/Encrypt" in data,
        "object_streams": data.count(b"/ObjStm"),
    }

    index = 0
    pos = 0
    total_streams = 0
    while True:
        s = data.find(b"stream", pos)
        if s < 0:
            break
        # ignore the word inside "endstream"
        if data[max(0, s - 3):s] == b"end":
            pos = s + 6
            continue
        total_streams += 1
        # data begins after CRLF / LF / CR following the keyword
        body = s + 6
        if data[body:body + 2] == b"\r\n":
            body += 2
        elif data[body:body + 1] in (b"\n", b"\r"):
            body += 1

        end = data.find(b"endstream", body)
        if end < 0:
            break
        # dictionary is the text just before the stream keyword
        dict_start = data.rfind(b"<<", max(0, s - 3000), s)
        header_blob = data[dict_start if dict_start >= 0 else max(0, s - 600):s]

        pos = end + 9
        if b"/FlateDecode" not in header_blob:
            continue
        # a filter chain that is not Flate-only means these bytes are not DEFLATE
        fm = re.search(rb"/Filter\s*(\[[^\]]*\]|/\w+)", header_blob)
        if fm:
            filters = re.findall(rb"/(\w+)", fm.group(1))
            if [f for f in filters if f != b"FlateDecode"]:
                report.other_method_entries += 1
                continue

        # prefer an explicit direct /Length
        blob = data[body:end]
        lm = re.search(rb"/Length\s+(\d+)(?!\s+\d+\s+R)", header_blob)
        if lm:
            declared = int(lm.group(1))
            if 0 < declared <= len(blob):
                blob = blob[:declared]
        else:
            blob = blob.rstrip(b"\r\n")

        if len(blob) < 3:
            continue
        # PDF Flate streams carry a zlib wrapper; skip CMF/FLG when present
        start_bit = 0
        meta = {"zlib_wrapped": False}
        if blob[0] & 0x0F == 8 and ((blob[0] << 8) | blob[1]) % 31 == 0:
            meta["zlib_wrapped"] = True
            meta["window_bits"] = (blob[0] >> 4) + 8
            meta["flevel"] = blob[1] >> 6
            payload = blob[2:]
        else:
            payload = blob

        obj_m = re.search(rb"(\d+)\s+(\d+)\s+obj", data[max(0, dict_start - 40):s])
        obj_id = obj_m.group(1).decode() if obj_m else str(index)
        subtype = None
        sm = re.search(rb"/Subtype\s*/(\w+)", header_blob)
        if sm:
            subtype = sm.group(1).decode("latin-1")
        tm = re.search(rb"/Type\s*/(\w+)", header_blob)
        meta.update(
            {
                "object": obj_id,
                "type": tm.group(1).decode("latin-1") if tm else None,
                "subtype": subtype,
                "offset": body,
            }
        )

        report.streams.append(
            DeflateStream(
                payload=payload,
                start_bit=start_bit,
                name=f"obj{obj_id}" + (f"/{subtype}" if subtype else ""),
                container="pdf",
                source=path,
                index=index,
                metadata=meta,
            )
        )
        index += 1

    if producer or creator:
        report.claims = {"producer": producer or creator, "source": "PDF /Producer"}
    report.metadata["total_stream_objects"] = total_streams
    report.metadata["flate_streams"] = len(report.streams)
    if not report.streams:
        raise ContainerError(
            "no /FlateDecode streams found (PDF may be encrypted, or use other filters)"
        )
    return report


def extract_raw(data: bytes, path: str) -> ContainerReport:
    report = ContainerReport(path=path, container="raw")
    report.streams.append(
        DeflateStream(payload=data, name=Path(path).name, container="raw", source=path)
    )
    return report


# --- entry point -----------------------------------------------------------


def extract_streams(
    path: str | Path, data: bytes | None = None, container: str | None = None
) -> ContainerReport:
    """Extract every DEFLATE stream from ``path``.

    ``data`` may be supplied to avoid a re-read; ``container`` forces a kind
    instead of sniffing it.
    """
    path = str(path)
    if data is None:
        data = Path(path).read_bytes()
    if not data:
        raise ContainerError("empty file")
    kind = container or detect_container(data, path)

    if kind in ("zip", "ooxml", "apk", "jar", "epub", "odf"):
        return extract_zip(data, path, kind)
    if kind == "gzip":
        return extract_gzip(data, path)
    if kind == "zlib":
        return extract_zlib(data, path)
    if kind == "png":
        return extract_png(data, path)
    if kind == "pdf":
        return extract_pdf(data, path)
    return extract_raw(data, path)
