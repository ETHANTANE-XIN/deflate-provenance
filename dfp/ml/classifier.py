"""High-level provenance classifier.

Wraps the random forest with:

* feature standardisation (mean/std from training);
* a temperature fitted on out-of-bag probabilities so the reported confidence
  is calibrated;
* an open-set novelty score -- the standardised distance to the nearest class
  centroid in whitened feature space -- so an unfamiliar stream can be rejected
  as ``unknown`` rather than misattributed;
* JSON-serialisable save/load (no pickle: the model is plain arrays).
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
    novelty: float
    proba: dict[str, float]
    top2_margin: float
    raw_top: str
    raw_confidence: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "abstained": self.abstained,
            "novelty": round(self.novelty, 3),
            "top2_margin": round(self.top2_margin, 4),
            "raw_top": self.raw_top,
            "raw_confidence": round(self.raw_confidence, 4),
            "proba": {k: round(v, 4) for k, v in sorted(
                self.proba.items(), key=lambda kv: -kv[1])},
        }


@dataclass
class ProvenanceClassifier:
    n_estimators: int = 60
    max_depth: int = 24
    seed: int = 12345
    min_confidence: float = 0.45
    min_margin: float = 0.08
    novelty_sigma: float = 8.0

    classes: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    _forest: RandomForest | None = None
    _mean: np.ndarray | None = None
    _std: np.ndarray | None = None
    _temperature: float = 1.0
    _class_centroids: np.ndarray | None = None
    _class_spread: np.ndarray | None = None
    _global_spread: float = 1.0

    # -- fitting ------------------------------------------------------------
    def fit(self, X, y_labels: list[str]) -> "ProvenanceClassifier":
        X = np.asarray(X, dtype=np.float64)
        self.classes = sorted(set(y_labels))
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[c] for c in y_labels], dtype=np.int64)

        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-9] = 1e-9
        Xs = (X - self._mean) / self._std

        self._forest = RandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            seed=self.seed,
        )
        self._forest.fit(Xs, y, len(self.classes))

        # calibrate temperature on OOB probabilities
        oob = self._forest.oob_proba(Xs)
        self._temperature = self._fit_temperature(oob, y)

        # class centroids + spread for novelty
        n_classes = len(self.classes)
        n_features = Xs.shape[1]
        cent = np.zeros((n_classes, n_features))
        spread = np.zeros(n_classes)
        for c in range(n_classes):
            rows = Xs[y == c]
            cent[c] = rows.mean(axis=0)
            d = np.linalg.norm(rows - cent[c], axis=1)
            spread[c] = d.std() if len(d) > 1 else 1.0
        spread[spread < 1e-6] = 1e-6
        self._class_centroids = cent
        self._class_spread = spread
        alld = np.linalg.norm(Xs - cent[y], axis=1)
        self._global_spread = float(alld.std()) or 1.0
        return self

    @staticmethod
    def _fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
        """Grid-search a temperature that minimises negative log-likelihood."""
        eps = 1e-9
        logp = np.log(np.clip(proba, eps, 1.0))
        best_t, best_nll = 1.0, float("inf")
        for t in np.linspace(0.5, 3.0, 26):
            scaled = logp / t
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            e = np.exp(scaled)
            p = e / e.sum(axis=1, keepdims=True)
            nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0)))
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        return best_t

    def _calibrate(self, proba: np.ndarray) -> np.ndarray:
        eps = 1e-9
        logp = np.log(np.clip(proba, eps, 1.0)) / self._temperature
        logp = logp - logp.max(axis=1, keepdims=True)
        e = np.exp(logp)
        return e / e.sum(axis=1, keepdims=True)

    # -- prediction ---------------------------------------------------------
    def predict(self, X) -> list[Prediction]:
        X = np.asarray(X, dtype=np.float64)
        Xs = (X - self._mean) / self._std
        raw = self._forest.predict_proba(Xs)
        cal = self._calibrate(raw)

        # novelty: two complementary terms, take the stronger.
        #  (a) per-class z-score: how unlike the predicted class's members;
        #  (b) global-distance ratio: how far from every class centroid at all
        #      (catches gross out-of-distribution streams a competent unknown
        #      encoder occasionally produces).
        diff = Xs[:, None, :] - self._class_centroids[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        z = dists / self._class_spread[None, :]
        global_ratio = dists.min(axis=1) / self._global_spread  # nearest centroid

        preds: list[Prediction] = []
        for i in range(len(Xs)):
            order = np.argsort(cal[i])[::-1]
            top = int(order[0])
            conf = float(cal[i, top])
            margin = float(cal[i, top] - cal[i, order[1]]) if len(order) > 1 else conf
            proba_dict = {self.classes[j]: float(cal[i, j]) for j in range(len(self.classes))}
            # novelty of the predicted class, combined with the global-distance
            # outlier term (stronger of the two)
            nov = float(max(z[i, top], global_ratio[i]))
            abstain = (
                nov > self.novelty_sigma
                or conf < self.min_confidence
                or margin < self.min_margin
            )
            preds.append(
                Prediction(
                    label=UNKNOWN if abstain else self.classes[top],
                    confidence=conf,
                    abstained=abstain,
                    novelty=nov,
                    proba=proba_dict,
                    top2_margin=margin,
                    raw_top=self.classes[top],
                    raw_confidence=conf,
                )
            )
        return preds

    def predict_one(self, vec) -> Prediction:
        return self.predict([vec])[0]

    def feature_importance(self) -> list[tuple[str, float]]:
        imp = self._forest.feature_importance(len(self.feature_names))
        pairs = list(zip(self.feature_names, imp.tolist()))
        return sorted(pairs, key=lambda kv: -kv[1])

    # -- persistence (plain JSON, no pickle) --------------------------------
    def save(self, path: str) -> None:
        blob = {
            "classes": self.classes,
            "feature_names": self.feature_names,
            "temperature": self._temperature,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "class_centroids": self._class_centroids.tolist(),
            "class_spread": self._class_spread.tolist(),
            "global_spread": self._global_spread,
            "min_confidence": self.min_confidence,
            "min_margin": self.min_margin,
            "novelty_sigma": self.novelty_sigma,
            "trees": [self._serialise_tree(t.root) for t in self._forest.trees],
            "n_classes": len(self.classes),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)

    @classmethod
    def load(cls, path: str) -> "ProvenanceClassifier":
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        obj = cls(
            min_confidence=blob["min_confidence"],
            min_margin=blob["min_margin"],
            novelty_sigma=blob["novelty_sigma"],
        )
        obj.classes = blob["classes"]
        obj.feature_names = blob["feature_names"]
        obj._temperature = blob["temperature"]
        obj._mean = np.array(blob["mean"])
        obj._std = np.array(blob["std"])
        obj._class_centroids = np.array(blob["class_centroids"])
        obj._class_spread = np.array(blob["class_spread"])
        obj._global_spread = blob["global_spread"]
        from .forest import DecisionTree

        forest = RandomForest()
        forest.n_classes = blob["n_classes"]
        forest.trees = []
        for t in blob["trees"]:
            tree = DecisionTree(blob["n_classes"])
            tree.root = obj._deserialise_tree(t)
            forest.trees.append(tree)
        forest.oob_indices = []
        obj._forest = forest
        return obj

    def _serialise_tree(self, node) -> dict:
        if node.proba is not None:
            return {"p": node.proba.tolist()}
        return {
            "f": node.feature,
            "t": node.threshold,
            "l": self._serialise_tree(node.left),
            "r": self._serialise_tree(node.right),
        }

    def _deserialise_tree(self, blob: dict):
        from .forest import _Node

        node = _Node()
        if "p" in blob:
            node.proba = np.array(blob["p"])
            return node
        node.feature = blob["f"]
        node.threshold = blob["t"]
        node.left = self._deserialise_tree(blob["l"])
        node.right = self._deserialise_tree(blob["r"])
        return node
