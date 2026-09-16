"""Per-stream analysis and per-archive vote aggregation.

A single archive (a .docx, an .apk) contains many DEFLATE streams.  Each stream
is attributed independently; the archive-level verdict aggregates those votes,
weighting each stream by how much evidence it carried (its uncompressed size,
capped) and by the classifier's confidence.

The aggregator also reports *internal consistency*: an archive whose streams
were all produced by one encoder is normal, while a mix of encoders is itself
forensically interesting (it suggests the archive was rebuilt or edited by a
different tool than it claims).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .containers import ContainerReport, DeflateStream, extract_streams
from .deflate import parse_stream, StreamRecord
from .features import extract_features, features_to_vector
from .ml import ProvenanceClassifier
from .ml.classifier import Prediction, UNKNOWN
from .signatures import SignatureHit, evaluate_signatures


@dataclass
class StreamAnalysis:
    name: str
    container: str
    out_size: int
    compressed_size: int
    n_blocks: int
    block_types: dict
    prediction: Prediction | None
    signatures: list[SignatureHit] = field(default_factory=list)
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "container": self.container,
            "out_size": self.out_size,
            "compressed_size": self.compressed_size,
            "n_blocks": self.n_blocks,
            "block_types": self.block_types,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "signatures": [
                {
                    "rule": s.rule,
                    "observation": s.observation,
                    "implication": s.implication,
                    "strength": s.strength,
                }
                for s in self.signatures
            ],
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class ArchiveVerdict:
    path: str
    container: str
    n_streams: int
    n_attributed: int
    n_abstained: int
    top_label: str
    top_share: float
    confidence: float
    vote_distribution: dict[str, float]
    consistent: bool
    metadata_summary: dict
    streams: list[StreamAnalysis] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self, include_streams: bool = True) -> dict:
        d = {
            "path": self.path,
            "container": self.container,
            "n_streams": self.n_streams,
            "n_attributed": self.n_attributed,
            "n_abstained": self.n_abstained,
            "top_label": self.top_label,
            "top_share": round(self.top_share, 4),
            "confidence": round(self.confidence, 4),
            "vote_distribution": {
                k: round(v, 4)
                for k, v in sorted(self.vote_distribution.items(), key=lambda kv: -kv[1])
            },
            "consistent": self.consistent,
            "metadata_summary": self.metadata_summary,
            "notes": self.notes,
        }
        if include_streams:
            d["streams"] = [s.to_dict() for s in self.streams]
        return d


def analyse_stream(
    stream: DeflateStream, clf: ProvenanceClassifier | None
) -> StreamAnalysis:
    try:
        record = parse_stream(stream.payload, start_bit=stream.start_bit, strict=False)
    except Exception as exc:  # pragma: no cover - defensive
        return StreamAnalysis(
            name=stream.label(), container=stream.container, out_size=0,
            compressed_size=stream.size, n_blocks=0, block_types={},
            prediction=None, error=str(exc), metadata=stream.metadata,
        )
    pred = None
    if clf is not None and record.n_tokens:
        vec = features_to_vector(extract_features(record))
        pred = clf.predict_one(vec)
    sigs = evaluate_signatures(record)
    return StreamAnalysis(
        name=stream.label(),
        container=stream.container,
        out_size=record.out_size,
        compressed_size=record.compressed_bytes,
        n_blocks=record.n_blocks,
        block_types=record.block_type_counts(),
        prediction=pred,
        signatures=sigs,
        error=record.error,
        metadata=stream.metadata,
    )


def aggregate_votes(analyses: list[StreamAnalysis]) -> tuple[dict[str, float], str, float]:
    """Size- and confidence-weighted vote over per-stream labels."""
    votes: dict[str, float] = defaultdict(float)
    for a in analyses:
        if not a.prediction or a.prediction.abstained:
            votes[UNKNOWN] += _evidence_weight(a.out_size) * 0.3
            continue
        weight = _evidence_weight(a.out_size) * a.prediction.confidence
        votes[a.prediction.label] += weight
    total = sum(votes.values()) or 1.0
    dist = {k: v / total for k, v in votes.items()}
    known = {k: v for k, v in dist.items() if k != UNKNOWN}
    if known:
        top = max(known, key=known.get)
        return dist, top, known[top]
    return dist, UNKNOWN, dist.get(UNKNOWN, 0.0)


def _evidence_weight(out_size: int) -> float:
    """More uncompressed bytes -> more evidence, with diminishing returns."""
    return math.log10(max(out_size, 1) + 10)


def analyse_archive(
    path: str,
    clf: ProvenanceClassifier | None,
    data: bytes | None = None,
    container: str | None = None,
) -> ArchiveVerdict:
    report: ContainerReport = extract_streams(path, data=data, container=container)
    analyses = [analyse_stream(s, clf) for s in report.streams]

    dist, top, top_share = aggregate_votes(analyses)
    attributed = [a for a in analyses if a.prediction and not a.prediction.abstained]
    abstained = [a for a in analyses if not a.prediction or a.prediction.abstained]

    labels = {a.prediction.label for a in attributed}
    consistent = len(labels) <= 1
    conf = (
        sum(a.prediction.confidence for a in attributed) / len(attributed)
        if attributed
        else 0.0
    )

    notes = list(report.notes)
    if not consistent:
        notes.append(
            f"streams attributed to multiple encoders ({sorted(labels)}): "
            "archive may have been rebuilt or partially edited"
        )
    _cross_check_metadata(report, top, notes)

    return ArchiveVerdict(
        path=path,
        container=report.container,
        n_streams=len(analyses),
        n_attributed=len(attributed),
        n_abstained=len(abstained),
        top_label=top,
        top_share=top_share,
        confidence=conf,
        vote_distribution=dict(dist),
        consistent=consistent,
        metadata_summary=report.metadata,
        streams=analyses,
        notes=notes,
    )


def _cross_check_metadata(report: ContainerReport, top_label: str, notes: list[str]) -> None:
    """Flag metadata claims that the bitstream evidence contradicts."""
    md = report.metadata
    hosts = md.get("host_systems") or []
    zlib_family_labels = {"zlib", "zlib_lineage", "java", "libarchive", "node", "dotnet"}
    # A Windows/Word-authored OOXML normally comes from the zlib lineage; an
    # attribution to a genuinely distinct implementation (dotnet, purepy, ...)
    # is what warrants a rebuilt-document flag.
    if (
        report.container == "ooxml"
        and top_label not in zlib_family_labels
        and top_label not in (UNKNOWN, "unknown")
    ):
        notes.append(
            f"OOXML container attributes to '{top_label}', not the zlib lineage "
            "typical of Microsoft Word; consistent with a rebuilt/regenerated document"
        )
    if md.get("prepended_bytes"):
        notes.append(
            f"{md['prepended_bytes']} bytes precede the ZIP structure "
            "(self-extracting stub or appended data)"
        )
