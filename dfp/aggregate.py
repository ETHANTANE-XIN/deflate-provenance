"""Per-entry attribution and the archive-level verdict (proposal III.A).

For a whole archive:

* every entry gets a status:
  ``attributed`` (a known profile, with confidence and setting),
  ``unknown encoder`` (the open-set rule fired), or
  ``insufficient evidence`` (stored without compression, or smaller than the
  minimum compressed size at which the model's held-out attribution was
  reliable);
* the per-entry results are combined by a **vote weighted by compressed
  size**; entries with insufficient evidence do not vote;
* two kinds of inconsistency are flagged:
  1. a profile that does not match the **producer the file claims**
     (``docProps/app.xml``, ODF ``meta.xml``, a JAR manifest), or ZIP
     compression-option bits that differ from those the claimed writer sets
     (see :mod:`dfp.producers`);
  2. **entries made by different encoders**, which suggests the archive was
     rebuilt or partly edited; the odd entries are named.

Because DEFLATE is lossless, a stream that was decompressed and recompressed
carries no trace of the earlier encoder, so the tool does not claim to detect
recompression itself; it reports these inconsistencies instead.

Before an entry is reported as made by a different encoder (or left as
"unknown"), it is re-encoded with the leading profile's reference encoder: an
exact byte match is proof that the archive's own encoder produced it, and
overrides the statistical result (so one misattributed or low-confidence part
neither raises a false alarm nor turns the verdict into "unknown").  As
further corroboration the largest entries are re-encoded with CPython zlib at
all 81 level/memory-level combinations (the brute-force baseline).
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass, field, replace

from .baselines import zlib_reencode_matches
from .containers import METHOD_DEFLATE, METHOD_STORE, ContainerReport, extract_streams
from .features import features_to_vector, stream_features
from .ml import ProvenanceClassifier
from .ml.classifier import UNKNOWN, Prediction
from .producers import Finding, check_producer, match_producer
from .signatures import SignatureHit, evaluate_signatures

ATTRIBUTED = "attributed"
UNKNOWN_ENCODER = "unknown encoder"
INSUFFICIENT = "insufficient evidence"
NOT_CLASSIFIED = "not classified"
ERROR = "error"


@dataclass
class EntryAnalysis:
    name: str
    container: str
    method: str
    status: str
    out_size: int = 0
    compressed_size: int = 0
    n_blocks: int = 0
    block_types: dict = field(default_factory=dict)
    prediction: Prediction | None = None
    signatures: list[SignatureHit] = field(default_factory=list)
    zlib_matches: list[tuple[int, int]] | None = None
    status_reason: str = ""
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.status == ATTRIBUTED and self.prediction:
            return self.prediction.label
        return self.status

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "container": self.container,
            "method": self.method,
            "status": self.status,
            "status_reason": self.status_reason,
            "out_size": self.out_size,
            "compressed_size": self.compressed_size,
            "n_blocks": self.n_blocks,
            "block_types": self.block_types,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "signatures": [
                {"rule": s.rule, "observation": s.observation,
                 "implication": s.implication, "strength": s.strength}
                for s in self.signatures
            ],
            "zlib_reencode_matches": (
                [f"level {lv} memLevel {ml}" for lv, ml in self.zlib_matches]
                if self.zlib_matches is not None else None
            ),
            "error": self.error,
            "metadata": self.metadata,
        }


# kept for callers of the earlier API
StreamAnalysis = EntryAnalysis


@dataclass
class ArchiveVerdict:
    path: str
    container: str
    profile: str
    share: float
    confidence: float
    setting: str | None
    vote_distribution: dict[str, float]
    entries: list[EntryAnalysis]
    findings: list[Finding]
    claims: dict
    metadata_summary: dict
    top_features: list[dict] = field(default_factory=list)
    model: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------
    def count(self, status: str) -> int:
        return sum(1 for e in self.entries if e.status == status)

    @property
    def inconsistent(self) -> bool:
        return any(f.kind == "inconsistent" for f in self.findings)

    @property
    def consistent(self) -> bool:
        return not self.inconsistent

    @property
    def top_label(self) -> str:  # earlier name of ``profile``
        return self.profile

    @property
    def streams(self) -> list[EntryAnalysis]:  # earlier name of ``entries``
        return self.entries

    def to_dict(self, include_entries: bool = True) -> dict:
        d = {
            "path": self.path,
            "container": self.container,
            "profile": self.profile,
            "share": round(self.share, 4),
            "confidence": round(self.confidence, 4),
            "setting": self.setting,
            "vote_distribution": {
                k: round(v, 4)
                for k, v in sorted(self.vote_distribution.items(), key=lambda kv: -kv[1])
            },
            "counts": {s: self.count(s) for s in (ATTRIBUTED, UNKNOWN_ENCODER, INSUFFICIENT,
                                                  NOT_CLASSIFIED, ERROR)},
            "inconsistent": self.inconsistent,
            "findings": [f.to_dict() for f in self.findings],
            "claims": self.claims,
            "top_features": self.top_features,
            "model": self.model,
            "metadata_summary": self.metadata_summary,
            "notes": self.notes,
        }
        if include_entries:
            d["entries"] = [e.to_dict() for e in self.entries]
        return d


def analyse_stream(stream, clf: ProvenanceClassifier | None, explain: int = 5) -> EntryAnalysis:
    record, feats = stream_features(stream.payload, stream.start_bit)
    entry = EntryAnalysis(
        name=stream.name or str(stream.index), container=stream.container, method="deflate",
        status=NOT_CLASSIFIED, out_size=record.out_size,
        compressed_size=record.compressed_bytes, n_blocks=record.n_blocks,
        block_types=record.block_type_counts(), signatures=evaluate_signatures(record),
        error=record.error, metadata=stream.metadata,
    )
    entry._output = record.output  # type: ignore[attr-defined]
    if record.error:
        entry.status = ERROR
        entry.status_reason = record.error
        return entry
    if clf is None:
        return entry
    if not record.n_tokens:
        entry.status = INSUFFICIENT
        entry.status_reason = "empty stream"
        return entry
    entry.prediction = clf.predict_one(features_to_vector(feats), explain=explain)
    if entry.compressed_size < clf.min_evidence_bytes:
        entry.status = INSUFFICIENT
        entry.status_reason = (
            f"{entry.compressed_size} compressed bytes, below the "
            f"{clf.min_evidence_bytes}-byte minimum at which held-out attribution "
            "was reliable")
    elif entry.prediction.abstained:
        entry.status = UNKNOWN_ENCODER
        entry.status_reason = entry.prediction.reason
    else:
        entry.status = ATTRIBUTED
    return entry


def aggregate_votes(entries: list[EntryAnalysis]) -> tuple[dict[str, float], str, float]:
    """Vote weighted by compressed size over entries with enough evidence."""
    votes: dict[str, float] = defaultdict(float)
    for e in entries:
        if e.status == ATTRIBUTED:
            votes[e.prediction.label] += e.compressed_size
        elif e.status == UNKNOWN_ENCODER:
            votes[UNKNOWN] += e.compressed_size
    total = sum(votes.values())
    if not total:
        return {}, INSUFFICIENT, 0.0
    dist = {k: v / total for k, v in votes.items()}
    known = {k: v for k, v in dist.items() if k != UNKNOWN}
    if known:
        top = max(known, key=known.get)
        if known[top] >= dist.get(UNKNOWN, 0.0):
            return dist, top, known[top]
    return dist, UNKNOWN, dist.get(UNKNOWN, 0.0)


def _top_features(entries: list[EntryAnalysis], profile: str, n: int = 6) -> list[dict]:
    acc: dict[str, float] = defaultdict(float)
    meaning: dict[str, str] = {}
    total = 0.0
    for e in entries:
        if e.status != ATTRIBUTED or e.prediction.label != profile:
            continue
        w = e.compressed_size
        total += w
        for item in e.prediction.explanation:
            acc[item["feature"]] += w * item["contribution"]
            meaning[item["feature"]] = item["meaning"]
    if not total:
        return []
    ranked = sorted(acc.items(), key=lambda kv: -abs(kv[1]))[:n]
    return [{"feature": k, "contribution": round(v / total, 4), "meaning": meaning[k]}
            for k, v in ranked]


def _top_attributed(entries: list[EntryAnalysis]) -> str | None:
    """Profile with the most compressed bytes among attributed entries."""
    weights: dict[str, float] = defaultdict(float)
    for e in entries:
        if e.status == ATTRIBUTED:
            weights[e.prediction.label] += e.compressed_size
    return max(weights, key=weights.get) if weights else None


def _payload(report: ContainerReport, entry: EntryAnalysis) -> bytes | None:
    for s in report.streams:
        if (s.name or str(s.index)) == entry.name:
            return s.payload[s.start_bit // 8 : s.start_bit // 8 + entry.compressed_size]
    return None


def reproduced_by(profile: str, raw: bytes, output: bytes | None) -> bool:
    """Does ``profile``'s reference encoder reproduce ``raw`` byte for byte?"""
    if output is None:
        return False
    if profile == "zlib":
        return bool(zlib_reencode_matches(raw, output))
    from .encoders import reference_for

    ref = reference_for(profile)
    if ref is None:
        return False
    try:
        return raw in ref.compress_many(output, ref.settings()).values()
    except Exception:
        return False


def analyse_archive(
    path: str,
    clf: ProvenanceClassifier | None,
    data: bytes | None = None,
    container: str | None = None,
    reencode: bool = True,
    reencode_limit: int = 8,
    explain: int = 5,
    verify: bool = True,
) -> ArchiveVerdict:
    report: ContainerReport = extract_streams(path, data=data, container=container)
    entries = [analyse_stream(s, clf, explain) for s in report.streams]

    # stored and non-DEFLATE ZIP entries are listed, never guessed at
    for e in report.entries:
        if e["method"] == METHOD_DEFLATE:
            continue
        stored = e["method"] == METHOD_STORE
        entries.append(EntryAnalysis(
            name=e["name"], container=report.container,
            method="stored" if stored else f"method {e['method']}",
            status=INSUFFICIENT,
            status_reason=("stored without compression: no DEFLATE data" if stored
                           else "not DEFLATE-compressed"),
            out_size=e["uncompressed_size"], compressed_size=e["compressed_size"],
        ))

    dist, profile, share = aggregate_votes(entries)
    verified: list[str] = []
    candidate = profile if profile not in (UNKNOWN, INSUFFICIENT) else _top_attributed(entries)
    if verify and candidate:
        # Before concluding, check the statistical results against proof: an
        # entry the forest places in another profile, or is unsure about, is
        # re-encoded with the leading profile's reference encoder.  An exact
        # byte match overrides the statistical result, so one misattributed or
        # low-confidence part neither raises a false "mixed" alarm nor drags
        # the archive verdict to "unknown".  An encoder that is genuinely
        # different can never pass this test.
        for e in entries:
            doubtful = (
                (e.status == ATTRIBUTED and e.prediction.label != candidate)
                or e.status == UNKNOWN_ENCODER
            )
            if not doubtful:
                continue
            raw = _payload(report, e)
            if raw is None or not reproduced_by(candidate, raw, getattr(e, "_output", None)):
                continue
            was = e.prediction.label if e.status == ATTRIBUTED else "unknown encoder"
            e.prediction = replace(e.prediction, label=candidate, abstained=False)
            e.status = ATTRIBUTED
            e.status_reason = (
                f"statistical result was '{was}', but the entry is exactly reproduced "
                f"by the '{candidate}' reference encoder, so it is counted as '{candidate}'")
            verified.append(e.name)
        if verified:
            dist, profile, share = aggregate_votes(entries)
    supporting = [e for e in entries if e.status == ATTRIBUTED and e.prediction.label == profile]
    weight = sum(e.compressed_size for e in supporting)
    confidence = (
        sum(e.prediction.confidence * e.compressed_size for e in supporting) / weight
        if weight else 0.0
    )
    setting = None
    if supporting:
        votes: dict[str, float] = defaultdict(float)
        for e in supporting:
            if e.prediction.setting:
                votes[e.prediction.setting] += e.compressed_size
        setting = max(votes, key=votes.get) if votes else None

    findings: list[Finding] = []
    # 1. entries made by different encoders
    attributed_labels = {e.prediction.label for e in entries if e.status == ATTRIBUTED}
    if profile in (UNKNOWN, INSUFFICIENT):
        # no archive-wide profile: entries disagree only if they name two or more
        others = ([e for e in entries if e.status == ATTRIBUTED]
                  if len(attributed_labels) > 1 else [])
    else:
        others = [e for e in entries if e.status == ATTRIBUTED and e.prediction.label != profile]
    if others and profile in (UNKNOWN, INSUFFICIENT):
        findings.append(Finding(
            "inconsistent", "mixed-encoders",
            f"attributed entries name {len(attributed_labels)} different profiles "
            f"({', '.join(sorted(attributed_labels))}): the archive may have been "
            "rebuilt or partly edited",
            entries=[e.name for e in others]))
    elif others:
        labels = sorted({e.prediction.label for e in others})
        findings.append(Finding(
            "inconsistent", "mixed-encoders",
            f"{len(others)} entr{'y was' if len(others) == 1 else 'ies were'} attributed to "
            f"{', '.join(labels)} while the archive as a whole matches '{profile}': "
            "the archive may have been rebuilt or partly edited",
            entries=[e.name for e in others]))
    elif clf is not None and profile not in (UNKNOWN, INSUFFICIENT):
        findings.append(Finding(
            "consistent", "mixed-encoders",
            f"all {len(supporting)} attributed entries match '{profile}'"
            + (f" ({len(verified)} confirmed by exact re-encoding after an uncertain "
               "or different statistical result)" if verified else "")))

    # 2. zlib re-encoding corroboration on the largest entries
    zlib_hits = 0
    checked = 0
    if reencode:
        deflate_entries = sorted(
            (e for e in entries if e.method == "deflate" and e.status != ERROR),
            key=lambda e: -e.compressed_size)[:reencode_limit]
        for e in deflate_entries:
            e.zlib_matches = zlib_reencode_matches(_payload(report, e),
                                                   getattr(e, "_output", None))
            checked += 1
            zlib_hits += bool(e.zlib_matches)
        if checked:
            findings.append(Finding(
                "info", "zlib-reencode",
                f"{zlib_hits} of the {checked} largest DEFLATE entries are reproduced "
                f"byte for byte by CPython zlib {zlib.ZLIB_RUNTIME_VERSION} (81 level/"
                "memLevel combinations tried)",
                entries=[e.name for e in deflate_entries if e.zlib_matches]))

    # 3. the claimed producer
    option_bits = report.metadata.get("option_bits_seen")
    classes = clf.classes if clf is not None else []
    if clf is not None:
        findings += check_producer(report.claims, profile if profile != INSUFFICIENT else UNKNOWN,
                                   classes, option_bits)
    producer = match_producer(report.claims.get("producer"))
    if producer and "zlib" in producer.excluded and zlib_hits:
        findings.append(Finding(
            "inconsistent", "producer",
            f"{report.claims.get('source', 'the container')} names {producer.label}, "
            f"yet {zlib_hits} entr{'y is' if zlib_hits == 1 else 'ies are'} exactly "
            f"reproduced by zlib, which {producer.label} is known not to produce "
            f"({producer.basis})"))

    notes = list(report.notes)
    if report.metadata.get("prepended_bytes"):
        notes.append(f"{report.metadata['prepended_bytes']} bytes precede the ZIP structure "
                     "(self-extracting stub or appended data)")

    model = {}
    if clf is not None:
        model = {
            "profiles": clf.classes,
            "min_confidence": clf.min_confidence,
            "max_distance": round(clf.max_distance, 3),
            "min_evidence_bytes": clf.min_evidence_bytes,
            "profile_versions": {
                k: v.get("reference_version") for k, v in clf.profile_info.items()
            },
        }
    return ArchiveVerdict(
        path=path,
        container=report.container,
        profile=profile,
        share=share,
        confidence=confidence,
        setting=setting,
        vote_distribution=dist,
        entries=entries,
        findings=findings,
        claims=report.claims,
        metadata_summary=report.metadata,
        top_features=_top_features(entries, profile),
        model=model,
        notes=notes,
    )
