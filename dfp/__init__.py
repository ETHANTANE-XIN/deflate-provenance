"""dfp -- DEFLATE Fingerprint Provenance.

A digital-forensics toolkit that attributes a compressed stream to the encoder
implementation that produced it, by reading the raw DEFLATE bitstream rather
than any container metadata.

Pipeline
--------
containers.py  locate DEFLATE streams inside ZIP/OOXML/APK/JAR/EPUB/GZIP/PNG/raw
deflate.py     bit-level RFC 1951 parser (no zlib used for parsing)
features.py    encoder-behaviour feature vector
signatures.py  deterministic hard rules
ml/            random forest + open-set rejection (pure numpy)
aggregate.py   per-stream votes -> per-archive verdict
report.py      JSON / HTML court-ready report
corpus.py      reproducible corpus generation over many encoders
adversarial.py metadata spoofing + recompression detection
covert.py      covert channel in encoder freedom, and its detector
evaluate.py    closed-set, minimum-evidence, open-set, adversarial evaluation
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
