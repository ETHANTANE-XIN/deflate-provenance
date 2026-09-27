# Bundled models

`default.json.gz` lets a fresh clone analyse files without training:

```
python -m dfp analyse suspicious.docx      # uses this model
python -m dfp.demo                         # so does the demo
python -m dfp models                       # shows it and any local models
```

* **Trained on:** the published reference corpus (`results/manifest.json`):
  348 source files (168 generated, 180 real files), 8,471 streams, 150 trees.
* **Profiles:** 7-Zip, Chromium zlib (Node.js), Go compress/flate,
  libdeflate, zlib (CPython, Java, .NET 8, libarchive), zlib-ng, zopfli, with
  the exact encoder versions in `results/manifest.json`.
* **Calibration** (a fifth of the source files held out): 92.7% accuracy;
  "unknown encoder" below 0.60 confidence or beyond distance 1.75;
  "insufficient evidence" below 256 compressed bytes.
* **Performance:** `results/evaluation.html` measures the same training
  procedure on a 70/30 split by source file.

**Not included:** Microsoft Word (no Word-saved documents were available).
Add them and train your own model; a model you train is saved in your local
store (`~/.dfp/models`, or `$DFP_MODELS`) and takes precedence over this one:

```
python -m dfp corpus -o corpus --sources <real files>
python -m dfp app corpus word_docs/ --name word
python -m dfp train --corpus corpus            # saved as your 'default'
```

A model records the feature set it was trained with. If the feature code
changes, loading it fails with a message to retrain; regenerate this file
with `python -m dfp train --corpus <reference corpus> -o dfp/bundled_models/default.json.gz`.
