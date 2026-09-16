"""Pure-numpy random forest for compressor attribution (class-weight aware)."""

from __future__ import annotations

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "proba")

    def __init__(self) -> None:
        self.feature: int = -1
        self.threshold: float = 0.0
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.proba: np.ndarray | None = None  # leaf class distribution


def _weighted_counts(y: np.ndarray, w: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.zeros(n_classes)
    np.add.at(counts, y, w)
    return counts


class DecisionTree:
    """CART classifier with weighted Gini impurity and random feature subsets."""

    def __init__(
        self,
        n_classes: int,
        max_depth: int = 24,
        min_samples_split: int = 4,
        max_features: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.root: _Node | None = None

    def _leaf(self, y: np.ndarray, w: np.ndarray) -> _Node:
        node = _Node()
        counts = _weighted_counts(y, w, self.n_classes)
        total = counts.sum()
        node.proba = counts / total if total else np.ones(self.n_classes) / self.n_classes
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, feats: np.ndarray):
        n = len(y)
        parent_counts = _weighted_counts(y, w, self.n_classes)
        W = parent_counts.sum()
        best_gain = 0.0
        best = None
        parent_gini = 1.0 - np.sum((parent_counts / W) ** 2)
        for f in feats:
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            col_sorted = col[order]
            y_sorted = y[order]
            w_sorted = w[order]
            left_counts = np.zeros(self.n_classes)
            right_counts = parent_counts.copy()
            wl = 0.0
            for i in range(1, n):
                cls = y_sorted[i - 1]
                wi = w_sorted[i - 1]
                left_counts[cls] += wi
                right_counts[cls] -= wi
                wl += wi
                if col_sorted[i] == col_sorted[i - 1]:
                    continue
                wr = W - wl
                if wl <= 0 or wr <= 0:
                    continue
                gl = 1.0 - np.sum((left_counts / wl) ** 2)
                gr = 1.0 - np.sum((right_counts / wr) ** 2)
                gain = parent_gini - (wl * gl + wr * gr) / W
                if gain > best_gain:
                    best_gain = gain
                    best = (f, (col_sorted[i] + col_sorted[i - 1]) / 2.0)
        return best, best_gain

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> "DecisionTree":
        self.root = self._build(X, y, w, 0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int) -> _Node:
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.all(y == y[0])
        ):
            return self._leaf(y, w)
        n_features = X.shape[1]
        k = self.max_features or max(1, int(np.sqrt(n_features)))
        feats = self.rng.choice(n_features, size=min(k, n_features), replace=False)
        best, gain = self._best_split(X, y, w, feats)
        if best is None or gain <= 0:
            return self._leaf(y, w)
        f, thr = best
        mask = X[:, f] <= thr
        if mask.all() or (~mask).all():
            return self._leaf(y, w)
        node = _Node()
        node.feature = int(f)
        node.threshold = float(thr)
        node.left = self._build(X[mask], y[mask], w[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], w[~mask], depth + 1)
        return node

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        node = self.root
        while node.proba is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.proba


class RandomForest:
    """Bootstrap-aggregated weighted decision trees with out-of-bag probabilities."""

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 24,
        min_samples_split: int = 4,
        max_features: int | None = None,
        class_weight: str | None = "balanced",
        seed: int = 12345,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weight = class_weight
        self.seed = seed
        self.trees: list[DecisionTree] = []
        self.oob_indices: list[np.ndarray] = []
        self.n_classes = 0

    def _weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight != "balanced":
            return np.ones(len(y))
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        inv = len(y) / (self.n_classes * counts)
        return inv[y]

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> "RandomForest":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes = n_classes
        n = len(y)
        base_w = self._weights(y)
        master = np.random.default_rng(self.seed)
        self.trees = []
        self.oob_indices = []
        for _ in range(self.n_estimators):
            rng = np.random.default_rng(master.integers(0, 2**63 - 1))
            idx = rng.integers(0, n, size=n)
            oob = np.setdiff1d(np.arange(n), np.unique(idx), assume_unique=False)
            tree = DecisionTree(
                n_classes, self.max_depth, self.min_samples_split,
                self.max_features, rng,
            )
            tree.fit(X[idx], y[idx], base_w[idx])
            self.trees.append(tree)
            self.oob_indices.append(oob)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros((len(X), self.n_classes))
        for i in range(len(X)):
            acc = np.zeros(self.n_classes)
            for tree in self.trees:
                acc += tree.predict_proba_one(X[i])
            out[i] = acc / len(self.trees)
        return out

    def oob_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        acc = np.zeros((n, self.n_classes))
        cnt = np.zeros(n)
        for tree, oob in zip(self.trees, self.oob_indices):
            for i in oob:
                acc[i] += tree.predict_proba_one(X[i])
                cnt[i] += 1
        cnt[cnt == 0] = 1
        return acc / cnt[:, None]

    def feature_importance(self, n_features: int) -> np.ndarray:
        counts = np.zeros(n_features)
        for tree in self.trees:
            stack = [tree.root]
            while stack:
                node = stack.pop()
                if node.proba is None:
                    counts[node.feature] += 1
                    stack.append(node.left)
                    stack.append(node.right)
        total = counts.sum()
        return counts / total if total else counts
