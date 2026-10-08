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
``endings``    how it ends every compressed part (see
               :data:`dfp.aggregate.ENDINGS`): "data" (a final block that
               carries data, as zlib's finish writes), "flush" (a sync flush
               and an empty final block) or "stored" (an empty stored final
               block).  This is structure, not statistics: it is checked on
               every entry without a model;
``basis``      where that knowledge comes from, printed in every report.

Entries are only as good as their basis, so every one records it.  Values
marked "measured" were measured with the tool on this project's reference
machine (see the README); the Microsoft Word values come from the team's
preliminary check described in the proposal.  Producers without verified
values carry ``None`` and are reported as "not yet profiled" rather than
guessed.  The team extends the table by profiling saved documents
(``python -m dfp app CORPUS DOCS --name NAME``) and adding an entry here.

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
    endings: frozenset[str] | None = None

    def matches(self, claim: str) -> bool:
        return re.search(self.pattern, claim, re.I) is not None


# Programs that compress through the zlib API get whichever library the
# platform provides: zlib, zlib-ng in compatibility mode (Fedora 40 and later)
# or Chromium's fork.
_ZLIB_API = frozenset({"zlib", "zlib-ng", "chromium-zlib"})
_DATA = frozenset({"data"})
_MEASURED = "measured with dfp on the reference machine"

PRODUCERS: list[Producer] = [
    Producer(
        "openpyxl", "openpyxl (Python)", r"openpyxl",
        expected=_ZLIB_API, option_bits=frozenset({0}), endings=_DATA,
        basis=f"{_MEASURED}: openpyxl 3.1.5 writes through Python's zipfile (the zlib "
              "API); 9/9 parts exactly reproduced by CPython zlib, each ending with a "
              "final block that carries data, option bits 'normal'",
    ),
    Producer(
        "libreoffice", "LibreOffice", r"LibreOffice",
        expected=_ZLIB_API, option_bits=frozenset({0}), endings=_DATA,
        basis=f"{_MEASURED}: LibreOffice 24.2.7.2 (Linux) writes through the zlib API; "
              "30/30 DOCX, XLSX and ODT files end every part with a final block that "
              "carries data, their parts are exactly reproduced by CPython zlib, and "
              "option bits are 'normal'",
    ),
    Producer(
        "microsoft-word", "Microsoft Word",
        r"^Microsoft (Office )?Word$|^Microsoft Macintosh Word$",
        # some Word builds compress parts exactly as zlib level 1 with a sync
        # flush does, so a zlib match is no evidence against Word
        expected=frozenset({"word", "zlib"}), option_bits=frozenset({3}),
        endings=frozenset({"flush"}),
        basis=f"{_MEASURED}: 60/60 documents saved by Word 16 (Windows) set 'super "
              "fast' on every entry and end every part with a sync flush and an empty "
              "final block.  Word builds differ in their match finder: the parts of "
              "these documents match no zlib setting, while other Word-saved files "
              "have parts that zlib level 1 with a sync flush reproduces exactly.  "
              "python-docx writes 'Microsoft Macintosh Word' while compressing with "
              "zlib's finish and 'normal' option bits (measured), which Word never "
              "does.  Mac builds of Word were not measured.",
    ),
    Producer(
        "microsoft-excel", "Microsoft Excel", r"^Microsoft (Office )?Excel$|^Microsoft Macintosh Excel$",
        basis="not yet profiled: add Excel-saved files with `python -m dfp app CORPUS DOCS --name excel`",
    ),
    Producer(
        "microsoft-powerpoint", "Microsoft PowerPoint",
        r"^Microsoft (Office )?PowerPoint$|^Microsoft Macintosh PowerPoint$",
        basis="not yet profiled: add PowerPoint-saved files with `python -m dfp app CORPUS DOCS --name powerpoint`",
    ),
    Producer(
        "jdk-jar", "JDK jar tool", r"^\d+(\.\d+)*\S*\s*\([^)]*\)$",
        expected=_ZLIB_API, option_bits=frozenset({0}), endings=_DATA,
        basis=f"{_MEASURED}: `jar` from OpenJDK 21.0.10 (java.util.zip, the zlib API); "
              "2/2 parts exactly reproduced by CPython zlib, each ending with a final "
              "block that carries data, option bits 'normal'",
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
    verdict_established: bool = True,
    zlib_proofs: int = 0,
) -> list[Finding]:
    """Compare the claimed producer with the stream evidence.

    A contradiction is only reported when the verdict is *established* (exact
    re-encoding, or confidence at the model's high-precision threshold);
    otherwise the claim can be neither confirmed nor contradicted.
    """
    if not verdict_established and verdict != "unknown":
        claim = claims.get("producer")
        producer = match_producer(claim) if claim else None
        if producer is not None and (verdict in producer.excluded or (
                producer.expected is not None and verdict not in producer.expected)):
            where = claims.get("source", "container")
            out = [Finding(
                "cannot-confirm", "producer",
                f"{where} names {producer.label}; the streams lean to '{verdict}', which "
                f"does not match it, but that attribution is neither confirmed by exact "
                "re-encoding nor high-confidence, so the claim is not contradicted")]
            return out + [f for f in check_producer(claims, verdict, model_profiles,
                                                    option_bits_seen, True, zlib_proofs)
                          if f.check == "option-bits"]
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
                f"files with `python -m dfp app CORPUS DOCS --name {sorted(producer.expected)[0]}`), and '{verdict}' is not excluded for it"))
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
