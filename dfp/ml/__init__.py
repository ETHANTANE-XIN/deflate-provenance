"""Classifier: a pure-NumPy random forest with calibrated confidence, an
open-set "unknown encoder" rule and traceable per-decision explanations.

scikit-learn is deliberately not required: a transparent, dependency-free
classifier the team controls end to end is easier to defend in a forensic
report than an opaque library call.  See :mod:`dfp.ml.classifier` for how the
proposal's calibration, rejection and explanation requirements are met.
"""

from __future__ import annotations

from .classifier import UNKNOWN, Prediction, ProvenanceClassifier
from .forest import RandomForest

__all__ = ["RandomForest", "ProvenanceClassifier", "Prediction", "UNKNOWN"]
