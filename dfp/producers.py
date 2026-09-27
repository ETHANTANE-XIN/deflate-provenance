"""Producer table: what each claimed producer is known to write.

Proposal section III.B: "a table maps each producer named in docProps/app.xml
to the profiles it is known to produce".  Each entry says, for a producer
named in the container (``docProps/app.xml`` ``Application``, ODF
``meta:generator``, a JAR manifest's ``Created-By``):

``expected``   profiles its streams are known to match;
``excluded``   profiles its streams are known *not* to match;
``option_bits`` the ZIP general-purpose-flag compression option (bits 1-2) it
               sets on DEFLATE entries (0 normal, 1 maximum, 2 fast,
               3 super fast);
``basis``      where that knowledge comes from, printed in every report.

Entries are only as good as their basis, so every one records it.  Values
marked "measured" were measured with the tool on this project's reference
machine (see the README); the Microsoft Word values come from the team's
preliminary check described in the proposal.  Producers without verified
values carry ``None`` and are reported as "not yet profiled" rather than
guessed.  The team extends the table by profiling saved documents
(``dfp app add``) and adding an entry here.

Order matters: more specific patterns (openpyxl's "Microsoft Excel
Compatible / Openpyxl") come before the generic ones they contain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Producer:
    key: str
    label: str
    pattern: str
    expected: frozenset[str] | None = None
    excluded: frozenset[str] = field(default_factory=frozenset)
    option_bits: frozenset[int] | None = None
    basis: str = ""

    def matches(self, claim: str) -> bool:
        return re.search(self.pattern, claim, re.I) is not None


_ZLIB = frozenset({"zlib"})
_MEASURED = "measured with dfp on the reference machine"

PRODUCERS: list[Producer] = [
    Producer(
        "openpyxl", "openpyxl (Python)", r"openpyxl",
        expected=_ZLIB, option_bits=frozenset({0}),
        basis=f"{_MEASURED}: openpyxl 3.1.5 writes through Python's zipfile; "
              "9/9 parts exactly reproduced by CPython zlib, option bits 'normal'",
    ),
    Producer(
        "libreoffice", "LibreOffice", r"LibreOffice",
        expected=_ZLIB, option_bits=frozenset({0}),
        basis=f"{_MEASURED}: LibreOffice 24.2.7.2 (Linux) DOCX, XLSX and ODT; "
              "25/25 parts exactly reproduced by CPython zlib, option bits 'normal'",
    ),
    Producer(
        "microsoft-word", "Microsoft Word",
        r"^Microsoft (Office )?Word$|^Microsoft Macintosh Word$",
        expected=frozenset({"word"}), excluded=_ZLIB, option_bits=frozenset({3}),
        basis="team's preliminary check (proposal section I): zlib reproduced none "
              "of the 38 compressed parts of two Word-saved documents under any of "
              "81 level/memLevel combinations, and Word set 'super fast' on every "
              "entry.  Note: python-docx also writes 'Microsoft Macintosh Word' "
              "here while compressing with zlib (measured).",
    ),
    Producer(
        "microsoft-excel", "Microsoft Excel", r"^Microsoft (Office )?Excel$|^Microsoft Macintosh Excel$",
        basis="not yet profiled: add Excel-saved files with `dfp app add`",
    ),
    Producer(
        "microsoft-powerpoint", "Microsoft PowerPoint",
        r"^Microsoft (Office )?PowerPoint$|^Microsoft Macintosh PowerPoint$",
        basis="not yet profiled: add PowerPoint-saved files with `dfp app add`",
    ),
    Producer(
        "jdk-jar", "JDK jar tool", r"^\d+(\.\d+)*\S*\s*\([^)]*\)$",
        expected=_ZLIB, option_bits=frozenset({0}),
        basis=f"{_MEASURED}: `jar` from OpenJDK 21.0.10 (java.util.zip, built on "
              "zlib); 2/2 parts exactly reproduced by CPython zlib, option bits 'normal'",
    ),
]


def match_producer(claim: str | None) -> Producer | None:
    if not claim:
        return None
    for producer in PRODUCERS:
        if producer.matches(claim.strip()):
            return producer
    return None


@dataclass
class Finding:
    """One consistency finding for the report."""

    kind: str  # "inconsistent" | "consistent" | "cannot-confirm" | "info"
    check: str  # "producer" | "option-bits" | "mixed-encoders" | ...
    text: str
    entries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "check": self.check, "text": self.text,
                "entries": self.entries}


def check_producer(
    claims: dict,
    verdict: str,
    model_profiles: list[str],
    option_bits_seen: list[int] | None,
) -> list[Finding]:
    """Compare the claimed producer with the stream evidence."""
    findings: list[Finding] = []
    claim = claims.get("producer")
    if not claim:
        findings.append(Finding("info", "producer",
                                "the container names no producer, so there is no claim to check"))
        return findings
    producer = match_producer(claim)
    where = claims.get("source", "container")
    if producer is None:
        findings.append(Finding(
            "cannot-confirm", "producer",
            f"{where} names '{claim}', which is not in the producer table; "
            "the stream attribution is reported without a claim to compare"))
        return findings

    if verdict == "unknown":
        findings.append(Finding(
            "cannot-confirm", "producer",
            f"{where} names {producer.label}; the streams match no known profile, "
            "so the claim can be neither confirmed nor contradicted"))
    elif verdict in producer.excluded:
        findings.append(Finding(
            "inconsistent", "producer",
            f"{where} names {producer.label}, but the compressed streams match the "
            f"'{verdict}' profile, which {producer.label} is known not to produce "
            f"({producer.basis}). This suggests the file was last written by a "
            f"'{verdict}'-based program rather than saved by {producer.label}."))
    elif producer.expected is not None and verdict not in producer.expected:
        known = sorted(p for p in producer.expected if p in model_profiles)
        if known:
            findings.append(Finding(
                "inconsistent", "producer",
                f"{where} names {producer.label}, whose output matches "
                f"{sorted(producer.expected)}, but the streams match '{verdict}'"))
        else:
            findings.append(Finding(
                "cannot-confirm", "producer",
                f"{where} names {producer.label}; its profile "
                f"{sorted(producer.expected)} is not in this model yet (add saved "
                f"files with `dfp app add`), and '{verdict}' is not excluded for it"))
    elif producer.expected is not None:
        findings.append(Finding(
            "consistent", "producer",
            f"{where} names {producer.label}, and the streams match '{verdict}' as "
            f"expected ({producer.basis})"))
    else:
        findings.append(Finding(
            "cannot-confirm", "producer",
            f"{where} names {producer.label}: {producer.basis}"))

    if producer.option_bits is not None and option_bits_seen:
        from .containers import OPTION_BITS

        seen = set(option_bits_seen)
        if not seen <= producer.option_bits:
            findings.append(Finding(
                "inconsistent", "option-bits",
                f"{producer.label} sets the ZIP compression option "
                f"{', '.join(OPTION_BITS[b] for b in sorted(producer.option_bits))!r} "
                f"on DEFLATE entries, but this file has "
                f"{', '.join(OPTION_BITS[b] for b in sorted(seen))!r}"))
        else:
            findings.append(Finding(
                "consistent", "option-bits",
                f"ZIP compression option bits ({', '.join(OPTION_BITS[b] for b in sorted(seen))}) "
                f"match what {producer.label} writes"))
    return findings
