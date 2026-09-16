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
    notes: list[str] = field(default_factory=list)


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


def detect_container(data: bytes, path: str | Path | None = None) -> str:
    """Identify the container from magic bytes, falling back to the suffix."""
    if data[:4] in (ZIP_LOCAL_SIG, ZIP_CENTRAL_SIG, ZIP_EOCD_SIG):
        if path:
            suffix = Path(path).suffix.lower()
            kind = CONTAINER_BY_SUFFIX.get(suffix)
            if kind and kind not in ("gzip", "png", "zlib", "raw"):
                return kind
        return "zip"
    if data[:2] == GZIP_MAGIC:
        return "gzip"
    if data[:8] == PNG_MAGIC:
        return "png"
    if data[:5] == PDF_MAGIC:
        return "pdf"
    if len(data) >= 2 and data[0] & 0x0F == 8 and ((data[0] << 8) | data[1]) % 31 == 0:
        return "zlib"
    if path:
        return CONTAINER_BY_SUFFIX.get(Path(path).suffix.lower(), "raw")
    return "raw"


# --- ZIP -------------------------------------------------------------------


def _find_eocd(data: bytes) -> int:
    """Locate the end-of-central-directory record, allowing for a comment."""
    limit = min(len(data), 65535 + 22)
    window = data[-limit:]
    idx = window.rfind(ZIP_EOCD_SIG)
    if idx < 0:
        raise ContainerError("no ZIP end-of-central-directory record found")
    return len(data) - limit + idx


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

    zip64 = ZIP64_EOCD_SIG in data[max(0, eocd - 64) : eocd]
    report.metadata.update(
        {
            "eocd_offset": eocd,
            "central_directory_offset": cd_offset,
            "central_directory_size": cd_size,
            "entry_count": entries_here,
            "archive_comment_len": comment_len,
            "archive_comment": comment[:200].decode("utf-8", "replace"),
            "zip64": zip64,
            "prepended_bytes": eocd - (cd_offset + cd_size) if cd_size else 0,
        }
    )

    pos = cd_offset
    order: list[str] = []
    host_systems: set[str] = set()
    versions: set[int] = set()
    extra_ids: set[str] = set()
    flags_seen: set[int] = set()
    timestamps: list[str] = []
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

        order.append(name)
        host = version_made >> 8
        host_systems.add(ZIP_HOST_SYSTEM.get(host, f"host{host}"))
        versions.add(version_made & 0xFF)
        flags_seen.add(flags)
        parsed_extra = _parse_extra(extra)
        extra_ids.update(f["name"] for f in parsed_extra)
        timestamps.append(_dos_datetime(mtime, mdate))

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
        if csize == 0 or csize == 0xFFFFFFFF:
            # data descriptor / zip64: fall back to the next signature
            end = data.find(ZIP_LOCAL_SIG, body)
            if end < 0:
                end = cd_offset
            payload = data[body:end]
        else:
            payload = data[body : body + csize]

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
            "deflate_entries": len(report.streams),
        }
    )
    return report


# --- GZIP / zlib / PNG / raw ----------------------------------------------


def extract_gzip(data: bytes, path: str) -> ContainerReport:
    report = ContainerReport(path=path, container="gzip")
    if data[:2] != GZIP_MAGIC:
        raise ContainerError("not a GZIP file")
    method = data[2]
    if method != METHOD_DEFLATE:
        raise ContainerError(f"GZIP compression method {method} is not DEFLATE")
    flags = data[3]
    mtime = int.from_bytes(data[4:8], "little")
    xfl = data[8]
    os_byte = data[9]
    pos = 10
    extra_meta: dict = {}
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
    if flags & 0x02:  # FHCRC
        pos += 2

    trailer = data[-8:] if len(data) >= pos + 8 else b""
    crc = int.from_bytes(trailer[:4], "little") if trailer else None
    isize = int.from_bytes(trailer[4:], "little") if trailer else None

    metadata = {
        "flags": flags,
        "mtime": mtime,
        "xfl": xfl,
        "xfl_meaning": {2: "best-compression", 4: "fastest"}.get(xfl, "unspecified"),
        "os_byte": os_byte,
        "os": GZIP_OS.get(os_byte, f"os{os_byte}"),
        "embedded_name": name,
        "header_len": pos,
        **extra_meta,
    }
    report.metadata = dict(metadata)
    report.streams.append(
        DeflateStream(
            payload=data[pos : len(data) - 8],
            name=name or Path(path).stem,
            container="gzip",
            source=path,
            declared_crc32=crc,
            declared_usize=isize,
            metadata=metadata,
        )
    )
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
