"""Profile classifier with calibrated confidence and open-set rejection.

Implements proposal section III.A ("How It Works"):

* a Random Forest compares a stream's features with the reference profiles;
* the reported confidence is **calibrated on held-out data**: a fifth of the
  training *source files* (never individual rows, so no source leaks) is kept
  aside, the forest is fitted on the rest, and a temperature is fitted on the
  held-out part;
* the classifier answers **"unknown encoder"** when that confidence falls
  below a threshold **or** the sample lies farther than a set distance from
  every known profile, so an encoder it was never trained on is rejected
  rather than misattributed (open-set recognition, Scheirer et al.).  Both
  thresholds are set from the held-out part, not guessed;
* each prediction is **traced through its trees** to report the features
  that contributed most (see :mod:`dfp.ml.forest`);
* a second, per-profile forest estimates the **setting** (for example the
  zlib level).  When several settings give identical output the training row
  carries the whole set, and any member counts as correct;
* the minimum compressed size at which held-out attribution is reliable is
  measured and stored, so the archive analyser can report smaller streams as
  "insufficient evidence" instead of guessing.

The model is saved as plain JSON (no pickle).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..features import FEATURE_NAMES
from .forest import RandomForest

UNKNOWN = "unknown"


@dataclass
class Prediction:
    label: str
    confidence: float
    abstained: bool
    reason: str
    distance: float
    proba: dict[str, float]
    raw_top: str
    setting: str | None = None
    setting_confidence: float | None = None
    explanation: list[dict] = field(default_factory=list)

    # the report layer used to call this "novelty"
    @property
    def novelty(self) -> float:
        return self.distance

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "abstained": self.abstained,
            "reason": self.reason,
            "distance": round(self.distance, 3),
            "raw_top": self.raw_top,
            "setting": self.setting,
            "setting_confidence": (
                round(self.setting_confidence, 4)
                if self.setting_confidence is not None else None
            ),
            "proba": {k: round(v, 4) for k, v in sorted(
                self.proba.items(), key=lambda kv: -kv[1]) if v >= 0.0005},
            "explanation": self.explanation,
        }


def _split_groups(groups: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Boolean mask selecting ``fraction`` of the distinct groups."""
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_hold = max(1, int(round(len(uniq) * fraction))) if len(uniq) > 1 else 0
    held = set(uniq[:n_hold].tolist())
    return np.array([g in held for g in groups.tolist()], dtype=bool)


@dataclass
class ProvenanceClassifier:
    n_estimators: int = 150
    max_depth: int = 24
    seed: int = 12345
    calibration_fraction: float = 0.2
    target_precision: float = 0.95
    distance_quantile: float = 0.99
    reliability_target: float = 0.90

    classes: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    min_confidence: float = 0.5
    max_distance: float = float("inf")
    min_evidence_bytes: int = 0
    temperature: float = 1.0
    calibration: dict = field(default_factory=dict)
    calibration_note: str = ""
    profile_info: dict = field(default_factory=dict)
    _forest: RandomForest | None = None
    _mean: np.ndarray | None = None
    _std: np.ndarray | None = None
    _centroids: np.ndarray | None = None
    _spread: np.ndarray | None = None
    _setting_models: dict = field(default_factory=dict)

    # -- helpers ---------------------------------------------------------------
    def _scale(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self._mean) / self._std, -10.0, 10.0)

    def _distances(self, Xs: np.ndarray) -> np.ndarray:
        """Distance to the nearest profile centroid, in units of that
        profile's own spread (1.0 = a typical member)."""
        diff = Xs[:, None, :] - self._centroids[None, :, :]
        d = np.linalg.norm(diff, axis=2) / self._spread[None, :]
        return d.min(axis=1)

    def _calibrate(self, proba: np.ndarray, t: float | None = None) -> np.ndarray:
        t = self.temperature if t is None else t
        logp = np.log(np.clip(proba, 1e-9, 1.0)) / t
        logp -= logp.max(axis=1, keepdims=True)
        e = np.exp(logp)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
        logp = np.log(np.clip(proba, 1e-9, 1.0))
        best_t, best_nll = 1.0, float("inf")
        for t in np.linspace(0.25, 4.0, 61):
            s = logp / t
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=1, keepdims=True)
            nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)))
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        return best_t

    # -- fitting ---------------------------------------------------------------
    def fit(
        self,
        X,
        labels: list[str],
        groups: list[str],
        settings: list[list[str]] | None = None,
        compressed_sizes: list[int] | None = None,
    ) -> "ProvenanceClassifier":
        """Fit on rows with one profile label each.

        ``groups`` are the source-file ids; the calibration hold-out is drawn
        by group.  ``settings`` (optional) gives each row's set of equivalent
        settings for the per-profile setting model.
        """
        X = np.asarray(X, dtype=np.float64)
        groups_a = np.asarray(groups)
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[c] for c in labels], dtype=np.int64)
        sizes = np.asarray(compressed_sizes if compressed_sizes is not None
                           else np.zeros(len(y)))

        hold = _split_groups(groups_a, self.calibration_fraction, self.seed)
        fit_mask = ~hold
        if hold.sum() == 0 or len(set(y[fit_mask].tolist())) < len(self.classes):
            fit_mask = np.ones(len(y), dtype=bool)
            hold = np.zeros(len(y), dtype=bool)

        self._mean = X[fit_mask].mean(axis=0)
        self._std = X[fit_mask].std(axis=0)
        self._std[self._std < 1e-9] = 1.0
        Xs = self._scale(X)

        self._forest = RandomForest(n_estimators=self.n_estimators,
                                    max_depth=self.max_depth, seed=self.seed)
        self._forest.fit(Xs[fit_mask], y[fit_mask], len(self.classes))

        cent = np.zeros((len(self.classes), Xs.shape[1]))
        spread = np.ones(len(self.classes))
        for c in range(len(self.classes)):
            rows = Xs[fit_mask & (y == c)]
            if len(rows):
                cent[c] = rows.mean(axis=0)
                d = np.linalg.norm(rows - cent[c], axis=1)
                spread[c] = max(float(np.sqrt(np.mean(d ** 2))), 1e-6)
        self._centroids, self._spread = cent, spread

        if hold.any():
            raw = self._forest.predict_proba(Xs[hold])
            yh = y[hold]
            self.temperature = self._fit_temperature(raw, yh)
            cal = self._calibrate(raw)
            conf = cal.max(axis=1)
            correct = cal.argmax(axis=1) == yh
            self.min_confidence = self._choose_confidence(conf, correct)
            dist = self._distances(Xs[hold])
            self.max_distance = float(np.quantile(dist, self.distance_quantile))
            answered = (conf >= self.min_confidence) & (dist <= self.max_distance)
            self.min_evidence_bytes = self._reliable_size(sizes[hold], correct, answered)
            self.calibration = {
                "held_out_rows": int(hold.sum()),
                "held_out_sources": int(len(set(groups_a[hold].tolist()))),
                "raw_accuracy": float(correct.mean()),
                "temperature": self.temperature,
                "min_confidence": self.min_confidence,
                "max_distance": self.max_distance,
                "min_evidence_bytes": self.min_evidence_bytes,
                "note": self.calibration_note,
                "target_precision": self.target_precision,
                "reliability_target": self.reliability_target,
            }
        else:
            self.calibration = {"held_out_rows": 0, "note": "too few sources to hold out"}

        # per-profile setting models (trained on all rows of the profile)
        self._setting_models = {}
        if settings is not None:
            for c, name in enumerate(self.classes):
                rows = np.where(y == c)[0]
                labels_c = [_canonical(settings[i]) for i in rows]
                distinct = sorted(set(labels_c))
                if len(distinct) < 2:
                    if distinct:
                        self._setting_models[name] = {"classes": distinct, "forest": None}
                    continue
                sy = np.array([distinct.index(l) for l in labels_c])
                rf = RandomForest(n_estimators=max(40, self.n_estimators // 3),
                                  max_depth=self.max_depth, seed=self.seed + c + 1)
                rf.fit(Xs[rows], sy, len(distinct))
                self._setting_models[name] = {"classes": distinct, "forest": rf}
        return self

    def _choose_confidence(self, conf: np.ndarray, correct: np.ndarray) -> float:
        """Smallest threshold whose accepted held-out rows reach the target
        precision (never below 0.4, so a coin-flip is never accepted)."""
        for t in np.linspace(0.4, 0.99, 60):
            keep = conf >= t
            if keep.sum() >= max(5, int(0.05 * len(conf))) and correct[keep].mean() >= self.target_precision:
                return float(round(t, 3))
        return 0.99

    #: compressed-size band edges used for the minimum-evidence measurement
    SIZE_EDGES = (0, 256, 1024, 4096, 16384, 65536)

    def _reliable_size(self, sizes: np.ndarray, correct: np.ndarray,
                       answered: np.ndarray | None = None) -> int:
        """Minimum compressed size for a reliable attribution.

        The smallest band edge whose own band of held-out data meets the
        reliability target (accuracy on the streams the model answers) *and*
        from which all larger streams, taken together, meet it too.  A dip in
        one larger band (for example very large streams where two encoders
        converge) is not a lack of evidence, so it must not push the minimum
        up.  Smaller streams are reported as "insufficient evidence".
        """
        if not len(sizes):
            return 0
        answered = np.ones(len(sizes), dtype=bool) if answered is None else answered
        edges = list(self.SIZE_EDGES) + [np.inf]
        for lo, hi in zip(edges[:-1], edges[1:]):
            band = (sizes >= lo) & (sizes < hi) & answered
            above = (sizes >= lo) & answered
            if band.sum() < 10:
                continue  # too little data to judge this band
            if (correct[band].mean() >= self.reliability_target
                    and correct[above].mean() >= self.reliability_target):
                return int(lo)
        self.calibration_note = "reliability target not reached at any size"
        return int(edges[-2])

    # -- prediction ------------------------------------------------------------
    def predict(self, X, explain: int = 0) -> list[Prediction]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Xs = self._scale(X)
        raw = self._forest.predict_proba(Xs)
        cal = self._calibrate(raw)
        dist = self._distances(Xs)
        top = cal.argmax(axis=1)
        contrib = None
        if explain:
            _, contrib = self._forest.contributions(Xs, top)

        preds: list[Prediction] = []
        for i in range(len(Xs)):
            conf = float(cal[i, top[i]])
            reasons = []
            if conf < self.min_confidence:
                reasons.append(f"confidence {conf:.2f} below {self.min_confidence:.2f}")
            if dist[i] > self.max_distance:
                reasons.append(
                    f"distance {dist[i]:.1f} beyond {self.max_distance:.1f} from every known profile")
            abstain = bool(reasons)
            name = self.classes[top[i]]
            setting = setting_conf = None
            if not abstain:
                setting, setting_conf = self._predict_setting(name, Xs[i])
            expl = []
            if contrib is not None:
                order = np.argsort(-np.abs(contrib[i]))[:explain]
                for j in order:
                    if abs(contrib[i, j]) < 1e-4:
                        continue
                    expl.append({
                        "feature": self.feature_names[j],
                        "value": float(X[i, j]),
                        "contribution": round(float(contrib[i, j]), 4),
                        "meaning": describe_feature(self.feature_names[j]),
                    })
            preds.append(Prediction(
                label=UNKNOWN if abstain else name,
                confidence=conf,
                abstained=abstain,
                reason="; ".join(reasons) if reasons else "attributed",
                distance=float(dist[i]),
                proba={self.classes[j]: float(cal[i, j]) for j in range(len(self.classes))},
                raw_top=name,
                setting=setting,
                setting_confidence=setting_conf,
                explanation=expl,
            ))
        return preds

    def predict_one(self, vec, explain: int = 5) -> Prediction:
        return self.predict([vec], explain=explain)[0]

    def _predict_setting(self, profile: str, xs: np.ndarray) -> tuple[str | None, float | None]:
        model = self._setting_models.get(profile)
        if not model:
            return None, None
        if model["forest"] is None:
            return model["classes"][0], 1.0
        p = model["forest"].predict_proba(xs[None, :])[0]
        j = int(np.argmax(p))
        return model["classes"][j], float(p[j])

    def feature_importance(self) -> list[tuple[str, float]]:
        imp = self._forest.feature_importance(len(self.feature_names))
        pairs = list(zip(self.feature_names, np.asarray(imp).tolist()))
        return sorted(pairs, key=lambda kv: -kv[1])

    # -- persistence -------------------------------------------------------------
    def save(self, path: str) -> None:
        blob = {
            "format": "dfp-model-2",
            "classes": self.classes,
            "feature_names": self.feature_names,
            "temperature": self.temperature,
            "min_confidence": self.min_confidence,
            "max_distance": self.max_distance,
            "min_evidence_bytes": self.min_evidence_bytes,
            "calibration": self.calibration,
            "profile_info": self.profile_info,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "centroids": self._centroids.tolist(),
            "spread": self._spread.tolist(),
            "forest": self._forest.to_dict(),
            "settings": {
                k: {"classes": v["classes"],
                    "forest": v["forest"].to_dict() if v["forest"] is not None else None}
                for k, v in self._setting_models.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)

    @classmethod
    def load(cls, path: str) -> "ProvenanceClassifier":
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        if blob.get("format") != "dfp-model-2":
            raise ValueError("model file is from an older dfp version; retrain with `dfp train`")
        if blob["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("model was trained on a different feature set; retrain it")
        obj = cls()
        obj.classes = blob["classes"]
        obj.feature_names = blob["feature_names"]
        obj.temperature = blob["temperature"]
        obj.min_confidence = blob["min_confidence"]
        obj.max_distance = blob["max_distance"]
        obj.min_evidence_bytes = blob["min_evidence_bytes"]
        obj.calibration = blob.get("calibration", {})
        obj.profile_info = blob.get("profile_info", {})
        obj._mean = np.array(blob["mean"])
        obj._std = np.array(blob["std"])
        obj._centroids = np.array(blob["centroids"])
        obj._spread = np.array(blob["spread"])
        obj._forest = RandomForest.from_dict(blob["forest"])
        obj._setting_models = {
            k: {"classes": v["classes"],
                "forest": RandomForest.from_dict(v["forest"]) if v["forest"] else None}
            for k, v in blob.get("settings", {}).items()
        }
        return obj


def _canonical(settings: list[str]) -> str:
    """Training label for a set of equivalent settings: prefer the reference
    program's own setting names (no ``program:`` prefix), then the first."""
    plain = [s for s in settings if ":" not in s]
    pool = plain or list(settings)
    return sorted(pool, key=_setting_key)[0]


def _setting_key(s: str):
    tail = s.split(":")[-1]
    digits = "".join(ch for ch in tail if ch.isdigit())
    return (0 if digits else 1, int(digits) if digits else 0, s)


_FEATURE_GROUPS = [
    ("dec_match_longest", "how often the encoder took the longest match available"),
    ("dec_match_shortfall", "how far chosen matches fell short of the longest available"),
    ("dec_shortfall_len", "match length at which the encoder stopped searching (nice length)"),
    ("dec_match_nearest", "whether the nearest copy of a match was chosen"),
    ("dec_match_farther", "how much farther than necessary chosen matches reached"),
    ("dec_min_match", "shortest match length the encoder emits"),
    ("dec_len3_far", "length-3 matches at distances over 4096 (zlib never emits these lazily)"),
    ("dec_lit_avail", "literals emitted although a match was available"),
    ("dec_lazy_defer", "literals followed by a longer match (lazy matching)"),
    ("dec_lazy_nogain", "deferred matches that were not longer"),
    ("dec_missed_match", "available matches the encoder skipped"),
    ("dec_avail3_far", "far length-3 matches the encoder declined"),
    ("dec_blk_tokens", "symbols per block (fixed buffer sizes show up here)"),
    ("dec_blk_boundary", "whether block splits paid for their extra header"),
    ("dec_blk_type", "whether the cheapest block type was chosen"),
    ("dec_blk_static", "cost of the dynamic tables relative to fixed codes"),
    ("huf_", "distance of the Huffman tables from the optimal ones"),
    ("blk_", "block-splitting policy"),
    ("cl_", "how the Huffman tables are encoded in the block header"),
    ("lit_", "literal/length Huffman table shape"),
    ("dst_", "distance Huffman table shape"),
    ("tok_", "literal and match mix"),
    ("len_", "match-length distribution"),
    ("dis_", "match-distance distribution"),
    ("str_", "stream-level artefacts (padding, size, density)"),
]


def describe_feature(name: str) -> str:
    for prefix, text in _FEATURE_GROUPS:
        if name.startswith(prefix):
            return text
    return name
