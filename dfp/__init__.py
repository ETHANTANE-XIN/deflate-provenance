"""dfp -- DeflateProvenance.

A digital-forensics tool that reads the DEFLATE bitstream inside ZIP, GZIP,
DOCX, XLSX, PPTX, APK and JAR files and estimates which known compressor
profile produced it, with a calibrated confidence or an "unknown encoder"
answer, and checks whether an archive's entries agree with each other and with
the producer the container claims.

Pipeline
--------
containers.py   locate DEFLATE streams, harvest metadata and producer claims
deflate.py      bit-level RFC 1951 parser (no zlib used for parsing)
features.py     statistical features of each stream
decisions.py    decision features: the encoder's choices versus the alternatives
corpus.py       reference corpus: every source x every encoder, verified labels
encoders/       zlib, Java, zlib-ng, Node (Chromium zlib), libdeflate, zopfli,
                7-Zip, Go, .NET, libarchive (+ the team's own pure-Python encoder)
ml/             random forest, held-out calibration, open-set rejection,
                per-decision explanations, setting inference
training.py     train a model from a corpus
aggregate.py    per-entry statuses, compressed-size vote, consistency findings
producers.py    producer table (claimed producer -> known profiles, option bits)
report.py       forensic HTML + JSON reports
baselines.py    preflate-rs, brute-force zlib re-encoding, metadata-only
zipwriter.py    lossless ZIP rewriting for the adversarial experiments
realfiles.py    application-saved documents (Word, LibreOffice) and their tests
evaluate.py     the proposal's evaluation plan
adversarial.py  metadata-robustness check, exact re-encoding test
covert.py       extension beyond the proposal: padding covert channel
"""

__version__ = "2.1.0"
__all__ = ["__version__"]
