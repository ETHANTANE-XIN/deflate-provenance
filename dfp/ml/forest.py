"""Pure-NumPy random forest (Breiman 2001) with traceable predictions.

Each tree is stored as flat arrays (feature, threshold, children, and the
class distribution at *every* node, not only the leaves).  Keeping the
internal-node distributions is what lets a prediction be traced through its
trees: walking a sample's path, the change in the predicted class's
probability at each split is credited to the feature that split used
(the decision-path method of Saabas).  Summed over the path, the credits plus
the root distribution equal the leaf probability exactly, so the explanation
adds up to the prediction.

Split search is vectorised: for each candidate feature the samples are
sorted once and the weighted Gini impurity of every split point is computed
from cumulative class weights in one pass.
"""

from __future__ import annotations

import numpy as np


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
        self.feature = np.zeros(0, dtype=np.int32)
        self.threshold = np.zeros(0)
        self.left = np.zeros(0, dtype=np.int32)
        self.right = np.zeros(0, dtype=np.int32)
        self.value = np.zeros((0, n_classes))
        self.importance: np.ndarray | None = None

    # -- fitting -------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> "DecisionTree":
        n_features = X.shape[1]
        self.importance = np.zeros(n_features)
        feats, thrs, lefts, rights, values = [], [], [], [], []
        onehot = np.eye(self.n_classes)[y] * w[:, None]
        total_w = w.sum()

        def new_node(counts: np.ndarray) -> int:
            s = counts.sum()
            values.append(counts / s if s > 0 else np.full(self.n_classes, 1 / self.n_classes))
            feats.append(-1)
            thrs.append(0.0)
            lefts.append(-1)
            rights.append(-1)
            return len(values) - 1

        root_counts = onehot.sum(axis=0)
        stack = [(np.arange(len(y)), 0, new_node(root_counts))]
        k = self.max_features or max(1, int(np.sqrt(n_features)))
        while stack:
            idx, depth, node = stack.pop()
            yi = y[idx]
            if (
                depth >= self.max_depth
                or len(idx) < self.min_samples_split
                or np.all(yi == yi[0])
            ):
                continue
            cand = self.rng.choice(n_features, size=min(k, n_features), replace=False)
            best = self._best_split(X[idx], onehot[idx], cand)
            if best is None:
                continue
            f, thr, gain = best
            mask = X[idx, f] <= thr
            if mask.all() or not mask.any():
                continue
            left_idx, right_idx = idx[mask], idx[~mask]
            node_w = onehot[idx].sum()
            self.importance[f] += gain * node_w / total_w
            feats[node] = int(f)
            thrs[node] = float(thr)
            lnode = new_node(onehot[left_idx].sum(axis=0))
            rnode = new_node(onehot[right_idx].sum(axis=0))
            lefts[node] = lnode
            rights[node] = rnode
            stack.append((left_idx, depth + 1, lnode))
            stack.append((right_idx, depth + 1, rnode))

        self.feature = np.array(feats, dtype=np.int32)
        self.threshold = np.array(thrs)
        self.left = np.array(lefts, dtype=np.int32)
        self.right = np.array(rights, dtype=np.int32)
        self.value = np.array(values)
        return self

    @staticmethod
    def _best_split(X: np.ndarray, onehot: np.ndarray, feats: np.ndarray):
        parent = onehot.sum(axis=0)
        W = parent.sum()
        if W <= 0:
            return None
        parent_gini = 1.0 - np.sum((parent / W) ** 2)
        best_gain = 1e-12
        best = None
        for f in feats:
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            cs = col[order]
            valid = cs[1:] != cs[:-1]
            if not valid.any():
                continue
            cum = np.cumsum(onehot[order], axis=0)[:-1]  # left counts after i
            wl = cum.sum(axis=1)
            wr = W - wl
            ok = valid & (wl > 0) & (wr > 0)
            if not ok.any():
                continue
            right = parent - cum
            with np.errstate(divide="ignore", invalid="ignore"):
                gl = 1.0 - np.sum((cum / wl[:, None]) ** 2, axis=1)
                gr = 1.0 - np.sum((right / wr[:, None]) ** 2, axis=1)
            gain = parent_gini - (wl * gl + wr * gr) / W
            gain[~ok] = -1.0
            i = int(np.argmax(gain))
            if gain[i] > best_gain:
                best_gain = float(gain[i])
                best = (int(f), float((cs[i] + cs[i + 1]) / 2.0), best_gain)
        return best

    # -- prediction ------------------------------------------------------------
    def apply(self, X: np.ndarray) -> np.ndarray:
        """Leaf index reached by every row of ``X``."""
        node = np.zeros(len(X), dtype=np.int32)
        active = np.arange(len(X))
        while len(active):
            f = self.feature[node[active]]
            internal = f >= 0
            active = active[internal]
            if not len(active):
                break
            nd = node[active]
            go_left = X[active, self.feature[nd]] <= self.threshold[nd]
            node[active] = np.where(go_left, self.left[nd], self.right[nd])
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.value[self.apply(X)]

    def contributions(self, X: np.ndarray, classes: np.ndarray, n_features: int) -> np.ndarray:
        """Per-feature change in P(class) along each row's decision path."""
        out = np.zeros((len(X), n_features))
        node = np.zeros(len(X), dtype=np.int32)
        rows = np.arange(len(X))
        active = rows.copy()
        while len(active):
            nd = node[active]
            f = self.feature[nd]
            internal = f >= 0
            active, nd, f = active[internal], nd[internal], f[internal]
            if not len(active):
                break
            go_left = X[active, f] <= self.threshold[nd]
            child = np.where(go_left, self.left[nd], self.right[nd])
            c = classes[active]
            delta = self.value[child, c] - self.value[nd, c]
            np.add.at(out, (active, f), delta)
            node[active] = child
        return out

    # -- persistence ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "feature": self.feature.tolist(),
            "threshold": self.threshold.tolist(),
            "left": self.left.tolist(),
            "right": self.right.tolist(),
            "value": np.round(self.value, 6).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict, n_classes: int) -> "DecisionTree":
        t = cls(n_classes)
        t.feature = np.array(d["feature"], dtype=np.int32)
        t.threshold = np.array(d["threshold"])
        t.left = np.array(d["left"], dtype=np.int32)
        t.right = np.array(d["right"], dtype=np.int32)
        t.value = np.array(d["value"])
        return t


class RandomForest:
    """Bootstrap-aggregated weighted decision trees."""

    def __init__(
        self,
        n_estimators: int = 150,
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
        self.n_classes = 0
        self.importance: np.ndarray | None = None

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
        importance = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            rng = np.random.default_rng(master.integers(0, 2**63 - 1))
            idx = rng.integers(0, n, size=n)
            tree = DecisionTree(n_classes, self.max_depth, self.min_samples_split,
                                self.max_features, rng)
            tree.fit(X[idx], y[idx], base_w[idx])
            importance += tree.importance
            self.trees.append(tree)
        total = importance.sum()
        self.importance = importance / total if total else importance
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        acc = np.zeros((len(X), self.n_classes))
        for tree in self.trees:
            acc += tree.predict_proba(X)
        return acc / max(len(self.trees), 1)

    def contributions(self, X: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(bias, contributions)`` for the given class of each row.

        ``bias[i] + contributions[i].sum()`` equals the forest's probability of
        ``classes[i]`` for row ``i``.
        """
        X = np.asarray(X, dtype=np.float64)
        classes = np.asarray(classes, dtype=np.int64)
        contrib = np.zeros((len(X), X.shape[1]))
        bias = np.zeros(len(X))
        for tree in self.trees:
            contrib += tree.contributions(X, classes, X.shape[1])
            bias += tree.value[0, classes]
        n = max(len(self.trees), 1)
        return bias / n, contrib / n

    def feature_importance(self, n_features: int | None = None) -> np.ndarray:
        """Mean decrease in (weighted) Gini impurity, normalised to sum to 1."""
        return self.importance if self.importance is not None else np.zeros(n_features or 0)

    def to_dict(self) -> dict:
        return {
            "n_classes": self.n_classes,
            "importance": self.importance.tolist() if self.importance is not None else None,
            "trees": [t.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RandomForest":
        forest = cls()
        forest.n_classes = d["n_classes"]
        forest.trees = [DecisionTree.from_dict(t, d["n_classes"]) for t in d["trees"]]
        imp = d.get("importance")
        forest.importance = np.array(imp) if imp is not None else None
        return forest
