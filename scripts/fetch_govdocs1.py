"""Fetch a Govdocs1 sample for the second test set (proposal section III.C).

Govdocs1 (Garfinkel et al., 2009) is published by Digital Corpora as 1,000
"thread" ZIP files of 1,000 documents each.  This script downloads the
requested threads, keeps the documents whose suffix is in ``--types`` and
writes them to ``--out``, ready for:

    python -m dfp evaluate --second-sources OUT          # as source files
    python -m dfp evaluate --real OUT                    # DOCX/XLSX/PPTX as real files

Usage:
    python scripts/fetch_govdocs1.py --threads 0 1 2 --out govdocs1_sample

The download needs outbound access to downloads.digitalcorpora.org, which the
environment that produced the published results did not have; ``--zip`` lets
you point at thread files you downloaded elsewhere.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://downloads.digitalcorpora.org/corpora/files/govdocs1/zipfiles/{:03d}.zip"


def extract(blob: bytes, out: Path, types: set[str], limit: int, taken: int) -> int:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            if taken >= limit:
                break
            suffix = Path(info.filename).suffix.lower().lstrip(".")
            if info.is_dir() or suffix not in types:
                continue
            target = out / Path(info.filename).name
            target.write_bytes(z.read(info))
            taken += 1
    return taken


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, nargs="+", default=[0])
    ap.add_argument("--zip", nargs="+", default=[], help="local thread ZIP files instead")
    ap.add_argument("--out", default="govdocs1_sample")
    ap.add_argument("--types", nargs="+",
                    default=["txt", "html", "xml", "csv", "pdf", "docx", "xlsx", "pptx"])
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    types = {t.lower() for t in args.types}
    taken = 0
    for path in args.zip:
        taken = extract(Path(path).read_bytes(), out, types, args.limit, taken)
    for t in args.threads:
        if taken >= args.limit or args.zip:
            break
        url = BASE.format(t)
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=600) as resp:
            taken = extract(resp.read(), out, types, args.limit, taken)
    print(f"{taken} files written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
