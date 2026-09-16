"""Classifier: a pure-numpy random forest with calibrated confidence and an
open-set abstain rule.

scikit-learn is not installed on the target machine, and for a forensic tool a
transparent, dependency-free classifier we control end to end is easier to
defend in a report than an opaque library call.  The implementation is a
standard CART random forest (Gini splits, bootstrap aggregation, random feature
subsets) plus:

* out-of-bag probability estimates used to fit a temperature so the reported
  confidence is calibrated rather than the forest's raw vote fraction;
* a per-class feature-space novelty score (standardised nearest-class
  distance) that lets an unseen encoder be rejected as ``unknown`` instead of
  being forced into a known class -- the forensic soundness property.
"""

from __future__ import annotations

from .forest import RandomForest
from .classifier import ProvenanceClassifier

__all__ = ["RandomForest", "ProvenanceClassifier"]
