"""Per-entry attribution and the archive-level verdict (proposal III.A).

For a whole archive:

* every entry gets a status:
  ``attributed`` (a known profile, with confidence and setting),
  ``unknown encoder`` (the open-set rule fired), or
  ``insufficient evidence`` (stored without compression, encrypted, or smaller
  than the model's minimum evidence size);
* **exact proof comes first.**  Every analysable entry is re-encoded with
  CPython zlib at all 81 level/memory-level combinations (and with a sync flush
  before finishing when the stream ends that way).  An exact byte match counts
  as proof only on an entry large enough for encoders to diverge
  (``PROOF_MIN_BYTES``) and with enough LZ77 decisions (``PROOF_MIN_MATCHES``):
  every encoder writes incompressible data the same way.  A proof labels the
  entry with confidence 1.0 and the matching setting, whatever the forest
  said.  Entries are also re-encoded with the reference encoders of the
  leading profile, of the model's most probable profiles (for entries matching
  no known profile) and of encoders proven elsewhere in the archive; which
  reference encoders were missing is recorded in the report, so a result never
  silently depends on the machine;
* the per-entry results are combined by a **vote weighted by compressed
  size**.  A profile is reported only when entries that match no known
  profile are not the majority, by count or by bytes; otherwise the verdict is
  ``unknown`` and the closest known profile is shown separately.  The archive
  confidence counts the bytes that do not support the verdict against it;
* two kinds of inconsistency are flagged, both on exact or structural
  evidence only (statistical disagreement alone is "cannot confirm"):
  1. **entries made by different writers.**  A writer ends every DEFLATE
     stream the same way (``ENDINGS``: a final block carrying data, a sync
     flush and an empty final block, or an empty stored final block), so
     entries that end differently were written by different programs or code
     paths; streams with almost no LZ77 matches are left out, since they carry
     no encoder decisions.  Entries proven by reference encoders of two
     different families, with no encoder reproducing both, show the same
     thing; libraries behind the zlib API (zlib, zlib-ng, Chromium's zlib)
     count as one family because they coincide with zlib on some inputs.  The
     entries named are those of the smaller side by entry count, since an edit
     touches few parts; a tie names none.  This is an inconsistency in ZIP,
     OOXML, ODF, JAR and EPUB files.  An APK is assembled by copying
     compressed entries from libraries unchanged, and in PDFs, PNGs and GZIP
     files streams from different writers are normal, so there it is
     reported as information;
  2. a **claimed producer** (``docProps/app.xml``, ODF ``meta.xml``, a JAR
     manifest) that the streams contradict: entries that do not end the way
     that producer ends every part, a profile it is known not to write
     (established by proof or high confidence), or ZIP compression-option
     bits it does not set (see :mod:`dfp.producers`).

Archives whose names repeat (two entries under one name) are reported as
inconsistent: legitimate writers never produce them, appending an edited part
does, and readers disagree about which copy they show.

Because DEFLATE is lossless, a stream that was decompressed and recompressed
carries no trace of the earlier encoder, so the tool does not claim to detect
recompression itself; it reports these inconsistencies instead.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass, field, replace

from .baselines import ZlibMatch, zlib_reencode_matches
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

#: compressed size from which an exact re-encoding match counts as proof.
#: Below it different encoders can coincide byte for byte (small parts of
#: genuine Word documents are reproduced by zlib with a sync flush).
PROOF_MIN_BYTES = 1024
#: LZ77 matches a stream needs before an exact match counts as proof: a stream
#: of stored blocks or (almost) only literals carries no match decisions, and
#: every encoder writes incompressible data the same way
PROOF_MIN_MATCHES = 64
#: matches a stream needs before its ending is compared with the others':
#: streams without match decisions (small images, audio) can take a different
#: code path inside one writer
ENDING_MIN_MATCHES = 8
#: how a DEFLATE stream ends.  One writer ends every stream the same way
#: (measured on every encoder and ZIP writer in the corpus, 60 Word documents,
#: 30 LibreOffice documents and over 1,500 APK, JAR, EPUB and ZIP files)
ENDINGS = {
    "data": "a final block that carries data (zlib's finish, and most encoders)",
    "flush": "a sync flush and an empty final block (as Microsoft Office writes)",
    "stored": "an empty stored final block (as Go's compress/flate writes)",
}
#: ZIP-based containers (the producer table describes these)
ARCHIVE_CONTAINERS = frozenset({"zip", "ooxml", "odf", "jar", "apk", "epub"})
#: containers whose entries one program compresses together, so mixed
#: encoders are an inconsistency.  An APK is assembled by copying compressed
#: entries from libraries unchanged, and in PDFs, PNGs and GZIP files streams
#: from different writers are normal: there it is information
MIXED_IS_INCONSISTENT = frozenset({"zip", "ooxml", "odf", "jar", "epub"})
#: libraries behind the zlib API that coincide with zlib on some inputs, so
#: their proofs do not show two different encoders
ZLIB_FAMILY = frozenset({"zlib", "zlib-ng", "chromium-zlib"})
#: reference encoders too slow to run on very large outputs
SLOW_REFERENCES = frozenset({"zopfli"})
SLOW_REFERENCE_MAX_OUTPUT = 256 * 1024
#: at most this many entries (largest first) are checked by exact re-encoding
REENCODE_MAX_ENTRIES = 400
#: entries per profile group that are re-encoded with the group's own
#: reference encoder to establish it
GROUP_PROOF_SAMPLES = 3
#: at most this many entries are re-encoded against another profile's
#: reference encoder (steps 2 and 3a)
REENCODE_SIDE_ENTRIES = 20
#: an entry matching no known profile is re-encoded with the reference
#: encoders of this many of the model's most probable profiles, each with at
#: least this probability
RESCUE_CANDIDATES = 3
RESCUE_MIN_PROBABILITY = 0.05
#: high-precision threshold used when a model does not record its own
DEFAULT_STRONG_CONFIDENCE = 0.9
#: a stream larger than this (compressed) is analysed on its first blocks up
#: to this size: far more than any training stream (sources are at most
#: 256 KB), and it keeps a 100 MB spreadsheet part from taking minutes
ANALYSE_MAX_BYTES = 1 << 20


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
    zlib_matches: list[ZlibMatch] | None = None
    status_reason: str = ""
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    #: position of this entry's stream in ``ContainerReport.streams``
    stream_index: int | None = None
    #: how the attribution was proven by exact re-encoding, if it was
    proof: str | None = None
    #: only the first ANALYSE_MAX_BYTES of a very large stream were analysed
    capped: bool = False
    #: how the stream ends (a key of ENDINGS), None when unknown
    ending: str | None = None
    #: LZ77 matches in the stream (the analysed part, when capped)
    n_matches: int = 0

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
            "proof": self.proof,
            "analysed_prefix_only": self.capped,
            "stream_ending": self.ending,
            "matches": self.n_matches,
            "signatures": [
                {"rule": s.rule, "observation": s.observation,
                 "implication": s.implication, "strength": s.strength}
                for s in self.signatures
            ],
            "zlib_reencode_matches": (
                [m.describe() for m in self.zlib_matches]
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
    #: nearest known profile when the verdict is "unknown"
    closest: str | None = None
    #: what exact verification could and could not run on this machine
    verification: dict = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    def count(self, status: str) -> int:
        return sum(1 for e in self.entries if e.status == status)

    @property
    def inconsistent(self) -> bool:
        return any(f.kind == "inconsistent" for f in self.findings)

    @property
    def status(self) -> str:
        """``inconsistent``, ``consistent`` or ``inconclusive`` for the report.

        "consistent" is only claimed when nothing could not be confirmed: an
        unknown verdict or any cannot-confirm finding makes it inconclusive.
        """
        if self.inconsistent:
            return "inconsistent"
        if self.profile in (UNKNOWN, INSUFFICIENT):
            return "inconclusive"
        if any(f.kind == "cannot-confirm" for f in self.findings):
            return "inconclusive"
        return "consistent"

    @property
    def consistent(self) -> bool:
        return self.status == "consistent"

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
            "closest_known_profile": self.closest,
            "status": self.status,
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
            "verification": self.verification,
            "metadata_summary": self.metadata_summary,
            "notes": self.notes,
        }
        if include_entries:
            d["entries"] = [e.to_dict() for e in self.entries]
        return d


def analyse_stream(stream, clf: ProvenanceClassifier | None, explain: int = 5,
                   stream_index: int | None = None,
                   max_bytes: int | None = ANALYSE_MAX_BYTES) -> EntryAnalysis:
    record, feats = stream_features(stream.payload, stream.start_bit, max_bytes=max_bytes)
    entry = EntryAnalysis(
        name=stream.name or str(stream.index), container=stream.container, method="deflate",
        status=NOT_CLASSIFIED, out_size=record.out_size,
        # a capped stream still votes with its full size
        compressed_size=stream.size if record.capped else record.compressed_bytes,
        n_blocks=record.n_blocks,
        block_types=record.block_type_counts(), signatures=evaluate_signatures(record),
        error=record.error, metadata=stream.metadata, stream_index=stream_index,
        capped=record.capped, ending=stream_ending(record), n_matches=record.n_matches,
    )
    entry._output = None if record.capped else record.output  # type: ignore[attr-defined]
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
            f"{entry.compressed_size} compressed bytes, below the model's "
            f"{clf.min_evidence_bytes}-byte minimum evidence size")
    elif entry.prediction.abstained:
        entry.status = UNKNOWN_ENCODER
        entry.status_reason = entry.prediction.reason
    else:
        entry.status = ATTRIBUTED
    if record.capped:
        entry.status_reason = (
            (entry.status_reason + "; " if entry.status_reason else "")
            + f"very large stream: its first {record.compressed_bytes} compressed bytes "
            f"({record.n_blocks} blocks) were analysed and exact re-encoding was skipped")
    return entry


def stream_ending(record) -> str | None:
    """How a parsed stream ends: "data", "flush" or "stored" (see ENDINGS).

    None when the stream has an error, was only partly decoded, or has no
    final block.  An empty final fixed block *without* a sync flush before it
    counts as "data": zlib writes one when its last block boundary happens to
    fall at the end of the input.
    """
    if record.error or record.capped or not record.blocks or not record.blocks[-1].bfinal:
        return None
    final = record.blocks[-1]
    prev = record.blocks[-2] if len(record.blocks) >= 2 else None
    if final.out_size == 0 and final.btype == 0:
        return "stored"
    if (final.out_size == 0 and prev is not None and prev.btype == 0
            and not prev.stored_len):
        return "flush"
    return "data"


def _family(profile: str) -> str:
    """Encoder family: libraries behind the zlib API count as one."""
    return "zlib-api" if profile in ZLIB_FAMILY else profile


def aggregate_votes(entries: list[EntryAnalysis]) -> tuple[dict[str, float], str, float]:
    """Vote weighted by compressed size over entries with enough evidence.

    Returns ``(distribution, verdict, share)``.  The verdict is the known
    profile with the most bytes, unless entries matching no known profile are
    the majority by count or by bytes, in which case it is ``unknown``.
    """
    votes: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for e in entries:
        if e.status == ATTRIBUTED:
            votes[e.prediction.label] += e.compressed_size
            counts[e.prediction.label] += 1
        elif e.status == UNKNOWN_ENCODER:
            votes[UNKNOWN] += e.compressed_size
            counts[UNKNOWN] += 1
    total = sum(votes.values())
    n = sum(counts.values())
    if not total:
        return {}, INSUFFICIENT, 0.0
    dist = {k: v / total for k, v in votes.items()}
    known = {k: v for k, v in dist.items() if k != UNKNOWN}
    unknown_bytes = dist.get(UNKNOWN, 0.0)
    unknown_count = counts.get(UNKNOWN, 0) / n
    if known and unknown_bytes <= 0.5 and unknown_count <= 0.5:
        top = max(known, key=known.get)
        return dist, top, known[top]
    return dist, UNKNOWN, unknown_bytes


def _top_features(entries: list[EntryAnalysis], profile: str, n: int = 6) -> list[dict]:
    acc: dict[str, float] = defaultdict(float)
    meaning: dict[str, str] = {}
    total = 0.0
    for e in entries:
        if e.status != ATTRIBUTED or e.prediction.label != profile or e.proof:
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
    """The entry's own compressed bytes, found by stream position (never by
    name: two entries can share a name)."""
    if entry.stream_index is None or not 0 <= entry.stream_index < len(report.streams):
        return None
    s = report.streams[entry.stream_index]
    return s.payload[s.start_bit // 8 : s.start_bit // 8 + entry.compressed_size]


def reproduced_by(profile: str, raw: bytes, output: bytes | None) -> bool:
    """Does ``profile``'s reference encoder reproduce ``raw`` byte for byte?"""
    return reproducing_setting(profile, raw, output) is not None


def reproducing_setting(profile: str, raw: bytes, output: bytes | None) -> str | None:
    """The reference-encoder setting of ``profile`` that reproduces ``raw``
    exactly, or ``None`` (also when the encoder is not installed)."""
    if output is None or raw is None:
        return None
    if profile == "zlib":
        matches = zlib_reencode_matches(raw, output)
        return _zlib_setting(matches) if matches else None
    from .encoders import reference_for

    ref = reference_for(profile)
    if ref is None:
        return None
    try:
        produced = ref.compress_many(output, ref.settings())
    except Exception:
        return None
    for setting, data in produced.items():
        if data == raw:
            return setting
    return None


def reference_available(profile: str) -> bool:
    if profile == "zlib":
        return True
    from .encoders import reference_for

    return reference_for(profile) is not None


def reference_exists(profile: str) -> bool:
    """Whether any reference adapter for ``profile`` exists, installed or not
    (application profiles such as Word have none)."""
    if profile == "zlib":
        return True
    from .encoders import list_encoders

    return any(e.library == profile and e.reference
               for e in list_encoders(only_available=False))


def _zlib_setting(matches: list[ZlibMatch]) -> str:
    """Setting label for a zlib proof: prefer the default memory level."""
    best = sorted(matches, key=lambda m: (m.mem_level != 8, m.level))[0]
    return f"{best.level}f" if best.sync_flush else str(best.level)


def _prove(entry: EntryAnalysis, label: str, setting: str | None, how: str) -> None:
    """Mark ``entry`` as exactly reproduced by ``label``'s encoder."""
    was = (entry.prediction.label if entry.status == ATTRIBUTED and entry.prediction
           else entry.status)
    base = entry.prediction or Prediction(label, 1.0, False, "", 0.0, {}, label)
    entry.prediction = replace(base, label=label, confidence=1.0, abstained=False,
                               setting=setting, setting_confidence=1.0,
                               reason="exact re-encoding")
    entry.status = ATTRIBUTED
    entry.proof = how
    if was != label:
        entry.status_reason = (f"statistical result was '{was}', but the entry is exactly "
                               f"reproduced by {how}")
    else:
        entry.status_reason = f"confirmed: exactly reproduced by {how}"


def analyse_archive(
    path: str,
    clf: ProvenanceClassifier | None,
    data: bytes | None = None,
    container: str | None = None,
    reencode: bool = True,
    reencode_limit: int = REENCODE_MAX_ENTRIES,
    explain: int = 5,
    verify: bool = True,
) -> ArchiveVerdict:
    report: ContainerReport = extract_streams(path, data=data, container=container)
    entries = [analyse_stream(s, clf, explain, i, max_bytes=ANALYSE_MAX_BYTES)
               for i, s in enumerate(report.streams)]

    # stored, encrypted and non-DEFLATE ZIP entries are listed, never guessed at
    for e in report.entries:
        if e.get("encrypted"):
            reason, method = "encrypted: the stored bytes are ciphertext, not DEFLATE", "encrypted"
        elif e["method"] == METHOD_DEFLATE:
            continue
        elif e["method"] == METHOD_STORE:
            reason, method = "stored without compression: no DEFLATE data", "stored"
        else:
            reason, method = "not DEFLATE-compressed", f"method {e['method']}"
        entries.append(EntryAnalysis(
            name=e["name"], container=report.container, method=method,
            status=INSUFFICIENT, status_reason=reason,
            out_size=e["uncompressed_size"], compressed_size=e["compressed_size"],
        ))

    strong = getattr(clf, "strong_confidence", None) or DEFAULT_STRONG_CONFIDENCE
    min_evidence = clf.min_evidence_bytes if clf is not None else 0
    proof_min = max(PROOF_MIN_BYTES, min_evidence)
    unavailable: set[str] = set()
    verification = {"zlib_checked": 0, "zlib_capped": 0, "reference_checks": 0,
                    "reference_encoders_unavailable": []}
    archive = report.container in ARCHIVE_CONTAINERS

    def provable(e: EntryAnalysis) -> bool:
        """Large enough, and with enough LZ77 decisions, for an exact match to
        identify an encoder.  A stream of stored blocks or (almost) only
        literals is written byte for byte alike by every encoder."""
        return e.compressed_size >= proof_min and e.n_matches >= PROOF_MIN_MATCHES

    available: dict[str, bool] = {}

    def ref_ok(label: str) -> bool:
        """The reference encoder of ``label`` is installed (asked once per file)."""
        if label not in available:
            available[label] = reference_available(label)
            if not available[label]:
                unavailable.add(label)
        return available[label]

    exact: dict[tuple[str, int], str | None] = {}

    def setting_of(label: str, e: EntryAnalysis) -> str | None:
        """The setting of ``label``'s reference encoder that reproduces ``e``
        byte for byte, or None.  Every (encoder, entry) pair runs at most once."""
        key = (label, id(e))
        if key in exact:
            return exact[key]
        out = getattr(e, "_output", None)
        found = None
        if out is None:
            pass  # very large stream: only its first blocks were decoded
        elif label == "zlib":
            if e.zlib_matches is None and reencode:
                e.zlib_matches = zlib_reencode_matches(_payload(report, e), out)
            found = _zlib_setting(e.zlib_matches) if e.zlib_matches else None
        elif verify and ref_ok(label) and not (label in SLOW_REFERENCES
                                               and len(out) > SLOW_REFERENCE_MAX_OUTPUT):
            verification["reference_checks"] += 1
            found = reproducing_setting(label, _payload(report, e), out)
        exact[key] = found
        return found

    def prove_with(label: str, e: EntryAnalysis) -> bool:
        setting = setting_of(label, e)
        if setting is None:
            return False
        _prove(e, label, setting, f"the '{label}' reference encoder (setting {setting})")
        return True

    # 1. exact zlib re-encoding: proof overrides the statistical result
    zlib_proven: list[EntryAnalysis] = []
    if reencode:
        def checkable(e: EntryAnalysis) -> bool:
            if e.method != "deflate" or e.status == ERROR or e.capped:
                return False
            # without a model every parsed entry is checked
            return clf is None or e.status != NOT_CLASSIFIED

        candidates = sorted((e for e in entries if checkable(e)), key=lambda e: -e.compressed_size)
        verification["zlib_capped"] = max(0, len(candidates) - reencode_limit)
        for e in candidates[:reencode_limit]:
            e.zlib_matches = zlib_reencode_matches(_payload(report, e), getattr(e, "_output", None))
            verification["zlib_checked"] += 1
            if e.zlib_matches and provable(e) and clf is not None:
                setting = _zlib_setting(e.zlib_matches)
                _prove(e, "zlib", setting, f"CPython zlib {zlib.ZLIB_RUNTIME_VERSION} "
                       f"({e.zlib_matches[0].describe()})")
                zlib_proven.append(e)

    dist, profile, share = aggregate_votes(entries)
    candidate = profile if profile not in (UNKNOWN, INSUFFICIENT) else _top_attributed(entries)

    # 2. doubtful entries against the leading profile's own reference encoder
    if verify and candidate and clf is not None and candidate != "zlib" and ref_ok(candidate):
        doubtful = sorted(
            (e for e in entries if provable(e) and (
                (e.status == ATTRIBUTED and e.prediction.label != candidate and not e.proof)
                or e.status == UNKNOWN_ENCODER)),
            key=lambda e: -e.compressed_size)[:REENCODE_SIDE_ENTRIES]
        if any([prove_with(candidate, e) for e in doubtful]):
            dist, profile, share = aggregate_votes(entries)

    # 2b. entries matching no known profile against the reference encoders of
    #     the model's leading candidates: a known encoder at a setting or on
    #     content unlike the training data is proven rather than left unknown
    if verify and clf is not None:
        rescued = False
        unknown_big = sorted((e for e in entries if e.status == UNKNOWN_ENCODER and provable(e)),
                             key=lambda e: -e.compressed_size)[:GROUP_PROOF_SAMPLES]
        for e in unknown_big:
            proba = e.prediction.proba if e.prediction else {}
            # zlib was already tried exactly in step 1
            leading = [p for p in sorted(proba, key=lambda p: -proba[p])
                       if p != "zlib" and proba[p] >= RESCUE_MIN_PROBABILITY]
            for label in leading[:RESCUE_CANDIDATES]:
                if reference_exists(label) and ref_ok(label) and prove_with(label, e):
                    rescued = True
                    break
        if rescued:
            dist, profile, share = aggregate_votes(entries)

    # 3. establish each profile group: exact proof on a few of its entries
    def regroup() -> dict[str, list[EntryAnalysis]]:
        out: dict[str, list[EntryAnalysis]] = defaultdict(list)
        for e in entries:
            if e.status == ATTRIBUTED:
                out[e.prediction.label].append(e)
        return out

    groups = regroup()
    if verify and clf is not None:
        for label, members in groups.items():
            if label == "zlib":
                continue  # every entry was tried against zlib in step 1
            if not reference_exists(label) or not ref_ok(label):
                continue
            # prove as many members as possible (each proof counts in step 5b);
            # give up on a label after GROUP_PROOF_SAMPLES misses in a row
            misses = 0
            for m in sorted((m for m in members if provable(m) and not m.proof),
                            key=lambda x: -x.compressed_size)[:REENCODE_SIDE_ENTRIES]:
                if prove_with(label, m):
                    misses = 0
                else:
                    misses += 1
                    if misses >= GROUP_PROOF_SAMPLES:
                        break

    # 3a. large unproven entries against the encoders proven in this archive:
    #     a part the model mislabelled is proven to belong with them
    proven_labels = sorted({e.prediction.label for e in entries if e.proof})
    if verify and clf is not None and proven_labels:
        moved = False
        unproven = sorted((e for e in entries if e.status == ATTRIBUTED and not e.proof
                           and provable(e)),
                          key=lambda e: -e.compressed_size)[:REENCODE_SIDE_ENTRIES]
        for e in unproven:
            for label in proven_labels:
                if label in ("zlib", e.prediction.label):
                    continue  # zlib was tried in step 1, the entry's own label in step 3
                if prove_with(label, e):
                    moved = True
                    break
        if moved:
            dist, profile, share = aggregate_votes(entries)
    groups = regroup()

    def established(label: str) -> bool:
        """Proven by exact re-encoding, or every member above the model's
        high-precision confidence with no exact zlib check against it."""
        members = groups.get(label, [])
        if not members:
            return False
        if any(m.proof for m in members):
            return True
        if not all(m.prediction.confidence >= strong for m in members):
            return False
        if sum(m.compressed_size for m in members) < PROOF_MIN_BYTES:
            return False  # too little evidence, however confident
        if label == "zlib":
            # zlib itself, at every setting, reproduces none of some member
            return not any(m.zlib_matches == [] for m in members)
        # zlib reproducing a member (even one too small for proof) is
        # evidence against any other label
        return not any(m.zlib_matches for m in members)

    # archive confidence: supporting evidence over all analysable bytes
    analysable = [e for e in entries if e.status in (ATTRIBUTED, UNKNOWN_ENCODER)]
    total_bytes = sum(e.compressed_size for e in analysable)
    closest = None
    if profile == UNKNOWN:
        closest = _top_attributed(entries)
    supporting = [e for e in entries if e.status == ATTRIBUTED and e.prediction.label == profile]
    confidence = (sum(e.prediction.confidence * e.compressed_size for e in supporting) / total_bytes
                  if total_bytes and supporting else 0.0)
    setting = None
    if supporting:
        svotes: dict[str, float] = defaultdict(float)
        for e in supporting:
            if e.prediction.setting:
                svotes[e.prediction.setting] += e.compressed_size
        setting = max(svotes, key=svotes.get) if svotes else None

    findings: list[Finding] = []
    n_unknown = sum(1 for e in entries if e.status == UNKNOWN_ENCODER)

    # 4. duplicate entry names
    dups = report.metadata.get("duplicate_names") or []
    if dups:
        findings.append(Finding(
            "inconsistent", "duplicate-names",
            f"{len(dups)} entry name(s) occur more than once. Legitimate writers never "
            "produce this; appending an edited copy of a part does, and readers disagree "
            "about which copy they show. Every copy is analysed separately.",
            entries=dups[:20]))

    # 5. entries made by different encoders.  Only exact evidence makes this
    #    an inconsistency: how the streams end (5a), or proofs by encoders of
    #    different families (5b).  Statistical disagreement alone is reported
    #    as "cannot confirm".
    mixed_kind = "inconsistent" if report.container in MIXED_IS_INCONSISTENT else "info"
    not_archive_note = (
        "" if mixed_kind == "inconsistent" else
        " An APK is assembled by copying compressed entries from libraries unchanged, so "
        "mixed encoders are normal there and not reported as an inconsistency."
        if report.container == "apk" else
        " In a PDF, PNG or multi-member GZIP, streams from different writers are normal "
        "(embedded images and fonts are passed through; members are concatenated), so this "
        "is not reported as an inconsistency.")

    # 5a. a writer ends every stream the same way
    ends: dict[str, list[EntryAnalysis]] = defaultdict(list)
    for e in entries:
        if e.method == "deflate" and e.ending and e.n_matches >= ENDING_MIN_MATCHES:
            ends[e.ending].append(e)

    # 5b. exact proofs by encoders of different families, each proof
    #     unambiguous: no encoder of the other family reproduces it too
    proofs: dict[str, list[EntryAnalysis]] = defaultdict(list)
    for e in entries:
        if e.proof and e.status == ATTRIBUTED:
            proofs[e.prediction.label].append(e)
    sides: dict[str, list[EntryAnalysis]] = defaultdict(list)
    if len({_family(p) for p in proofs}) >= 2:
        for label, members in proofs.items():
            rivals = [o for o in proofs if _family(o) != _family(label)]
            for m in members:
                if not any(setting_of(o, m) for o in rivals):
                    sides[_family(label)].append(m)
    if len(sides) < 2:
        sides = defaultdict(list)
    else:
        # entries too small (or with too few matches) for proof still show
        # which side they are on when exactly one of the proven encoders
        # reproduces them: the archive is already proven mixed, so this only
        # decides which side is the odd one
        reps = {fam: next(p for p in proofs if _family(p) == fam) for fam in sides}
        placed = {id(m) for ms in sides.values() for m in ms}
        loose = sorted((e for e in entries if e.method == "deflate" and id(e) not in placed
                        and e.status in (ATTRIBUTED, UNKNOWN_ENCODER, INSUFFICIENT)
                        and e.n_matches >= ENDING_MIN_MATCHES),
                       key=lambda e: -e.compressed_size)[:REENCODE_SIDE_ENTRIES]
        for e in loose:
            hits = [fam for fam, label in reps.items() if setting_of(label, e)]
            if len(hits) == 1:
                sides[hits[0]].append(e)

    def odd_text(counts: dict[str, int]) -> tuple[list[str], bool]:
        most = max(counts.values())
        odd = sorted(k for k, c in counts.items() if c < most)
        return odd, not odd

    if len(ends) >= 2:
        odd, tied = odd_text({k: len(v) for k, v in ends.items()})
        how = "; ".join(f"{len(ends[k])} end{'s' if len(ends[k]) == 1 else ''} with "
                        f"{ENDINGS[k]}" for k in sorted(ends, key=lambda k: -len(ends[k])))
        findings.append(Finding(
            mixed_kind, "mixed-encoders",
            f"the entries' DEFLATE streams end in different ways ({how}). A writer ends "
            "every stream the same way, so at least two writers made this file: it may "
            "have been rebuilt or partly edited. "
            + ("Equal numbers of entries end each way, so which entries were changed "
               "cannot be told." if tied else
               f"The odd entries are those that end with {' or '.join(ENDINGS[k] for k in odd)}.")
            + not_archive_note,
            entries=[m.name for k in odd for m in ends[k]]))
    elif sides:
        odd, tied = odd_text({k: len(v) for k, v in sides.items()})
        how = "; ".join(f"{len(v)} entr{'y' if len(v) == 1 else 'ies'} reproduced exactly by "
                        f"{', '.join(sorted({m.prediction.label for m in v if m.proof}))}"
                        for k, v in sorted(sides.items(), key=lambda kv: -len(kv[1])))
        findings.append(Finding(
            mixed_kind, "mixed-encoders",
            f"entries were made by different encoders ({how}; no encoder reproduces "
            "entries of both kinds). The archive may have been rebuilt or partly edited. "
            + ("The sides have equal numbers of entries, so which entries were changed "
               "cannot be told." if tied else
               f"The odd entries are those reproduced by "
               f"{', '.join(sorted({m.prediction.label for k in odd for m in sides[k] if m.proof}))}.")
            + not_archive_note,
            entries=[m.name for k in odd for m in sides[k]]))
    elif len(groups) > 1:
        bytes_of = {g: sum(m.compressed_size for m in ms) for g, ms in groups.items()}
        summary = "; ".join(
            f"{g}: {len(groups[g])} entr{'y' if len(groups[g]) == 1 else 'ies'}, "
            f"{bytes_of[g] / max(total_bytes, 1):.0%} of the bytes"
            + (" (proven)" if any(m.proof for m in groups[g]) else
               " (high confidence)" if established(g) else "")
            for g in sorted(groups, key=lambda g: -bytes_of[g]))
        findings.append(Finding(
            "cannot-confirm", "mixed-encoders",
            f"entries lean to different profiles ({summary}), but neither how the "
            "streams end nor exact re-encoding by two different encoder families shows "
            "two writers. The model's disagreement alone is not reported as an edit: it "
            "is typical of one encoder the model does not know, or of zlib-family "
            "libraries, which coincide with zlib on some inputs."))
    elif groups and profile not in (UNKNOWN, INSUFFICIENT):
        proven = sum(1 for e in supporting if e.proof)
        findings.append(Finding(
            "consistent", "mixed-encoders",
            f"all {len(supporting)} attributed entries match '{profile}'"
            + (f" ({proven} confirmed by exact re-encoding)" if proven else "")
            + (f"; {n_unknown} other entr{'y matches' if n_unknown == 1 else 'ies match'} "
               "no known profile" if n_unknown else "")
            + (f", and every stream ends with {ENDINGS[next(iter(ends))]}" if len(ends) == 1 else "")))
    elif profile == UNKNOWN:
        findings.append(Finding(
            "cannot-confirm", "mixed-encoders",
            f"{n_unknown} of {len(analysable)} analysable entries match no known profile"
            + (f"; the closest known profile is '{closest}'" if closest else "")
            + ", so whether the entries share one encoder cannot be judged"))

    # 6. exact zlib re-encoding corroboration
    checked = [e for e in entries if e.zlib_matches is not None]
    if checked:
        hits = [e for e in checked if e.zlib_matches]
        big = [e for e in hits if provable(e)]
        findings.append(Finding(
            "info", "zlib-reencode",
            f"{len(hits)} of the {len(checked)} entries checked are reproduced byte for byte "
            f"by CPython zlib {zlib.ZLIB_RUNTIME_VERSION} (81 level/memLevel combinations, "
            f"with and without a sync flush); {len(big)} of them are at least {proof_min} "
            f"compressed bytes with at least {PROOF_MIN_MATCHES} matches, from which a match "
            "counts as proof"
            + (f". {verification['zlib_capped']} smaller entries were not checked"
               if verification["zlib_capped"] else ""),
            entries=[e.name for e in big][:20]))

    # 7. the claimed producer (the producer table describes ZIP-based formats)
    option_bits = report.metadata.get("option_bits_seen")
    classes = clf.classes if clf is not None else []
    claim = report.claims.get("producer")
    producer = match_producer(claim) if archive else None
    where = report.claims.get("source", "the container")
    if archive:
        verdict_for_claim = profile if clf is not None and profile != INSUFFICIENT else UNKNOWN
        findings += check_producer(
            report.claims, verdict_for_claim, classes, option_bits,
            verdict_established=(established(verdict_for_claim)
                                 if verdict_for_claim != UNKNOWN else False))
    elif claim:
        findings.append(Finding(
            "info", "producer",
            f"{where} names '{claim}'; the producer table describes ZIP-based formats "
            "(OOXML, ODF, JAR), so this claim is not checked"))
    # how the claimed producer ends its streams: structure, checked exactly
    if producer and producer.endings and ends:
        off = [m for k, ms in ends.items() if k not in producer.endings for m in ms]
        if off:
            n_ends = sum(len(ms) for ms in ends.values())
            findings = [f for f in findings
                        if not (f.check == "producer" and f.kind in ("consistent", "cannot-confirm"))]
            findings.append(Finding(
                "inconsistent", "producer",
                f"{where} names {producer.label}, which ends every compressed part with "
                f"{' or '.join(ENDINGS[k] for k in sorted(producer.endings))} "
                f"({producer.basis}); {len(off)} of {n_ends} entries end otherwise, so they "
                f"were not written by {producer.label}",
                entries=[m.name for m in off][:20]))
    if report.claims.get("source_copies"):
        findings.append(Finding(
            "info", "producer",
            f"{report.claims['source']} occurs {report.claims['source_copies']} times; the "
            "claim was read from the first copy in central-directory order"))

    # 8. what could not be verified on this machine
    no_encoder = sorted(p for p in unavailable if not reference_exists(p))
    missing = sorted(unavailable - set(no_encoder))
    verification["reference_encoders_unavailable"] = missing
    verification["profiles_without_reference_encoder"] = no_encoder
    if missing:
        findings.append(Finding(
            "info", "verification",
            "exact re-encoding with the reference encoder of "
            f"{', '.join(missing)} was not possible on this machine (not "
            "installed), so attributions to those profiles rest on the statistical model "
            "alone; `dfp encoders` lists what is installed"))
    if no_encoder:
        findings.append(Finding(
            "info", "verification",
            f"{', '.join(no_encoder)} {'is an application profile' if len(no_encoder) == 1 else 'are application profiles'} "
            "learnt from saved files: there is no reference encoder to re-encode with, so "
            "attributions to it rest on the statistical model and on exact checks against "
            "the other encoders"))
    if report.encrypted_entries:
        findings.append(Finding(
            "info", "encrypted",
            f"{report.encrypted_entries} encrypted entr"
            f"{'y was' if report.encrypted_entries == 1 else 'ies were'} not analysed: "
            "their bytes are ciphertext"))
    if report.metadata.get("backslash_names"):
        findings.append(Finding(
            "info", "entry-names",
            f"{report.metadata['backslash_names']} entry name(s) use '\\' separators, which "
            "APPNOTE 4.4.17 forbids; Windows PowerShell 5.1 Compress-Archive writes them"))

    notes = list(report.notes)

    model = {}
    if clf is not None:
        model = {
            "profiles": clf.classes,
            "min_confidence": clf.min_confidence,
            "strong_confidence": strong,
            "max_distance": round(clf.max_distance, 3),
            "novelty": getattr(clf, "novelty_info", {}) or {},
            "min_evidence_bytes": clf.min_evidence_bytes,
            "min_evidence_floor": getattr(clf, "min_evidence_floor", None),
            "proof_min_bytes": proof_min,
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
        top_features=_top_features(entries, profile if profile != UNKNOWN else (closest or "")),
        model=model,
        notes=notes,
        closest=closest,
        verification=verification,
    )
