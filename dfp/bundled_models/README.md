# Bundled models

`default.json.gz` lets a fresh clone analyse files without training:

```
python -m dfp analyse suspicious.docx      # uses this model
python -m dfp.demo                         # so does the demo
python -m dfp models                       # shows it and any local models
```

* **Trained on:** the published reference corpus (`results/manifest.json`):
  286 source files (168 generated, 118 real files), plus the training half of
  60 documents saved by Microsoft Word 16 on Windows; 8,782 streams from 346
  source files, 150 trees. Trained by DeflateProvenance 2.1.0 with
  `scripts/reproduce.sh` in the reference image.
* **Profiles (8):** 7-Zip, Chromium zlib (Node.js), Go compress/flate,
  libdeflate, Microsoft Word 16 (an application profile learnt from saved
  documents), zlib (CPython, Java, .NET 8, libarchive), zlib-ng, zopfli, with
  the exact encoder versions in `results/manifest.json`.
* **Calibration** (a fifth of the source files held out, 69 files): 95.2%
  accuracy; "unknown encoder" below 0.40 confidence or beyond novelty 4.95;
  high-precision confidence 0.89; "insufficient evidence" below 256
  compressed bytes.
* **Performance:** `results/evaluation.html` measures the same training
  procedure on a 70/30 split by source family.

A model you train is saved in your local store (`~/.dfp/models`, or
`$DFP_MODELS`) and takes precedence over this one:

```
python -m dfp corpus -o corpus --sources <real files>
python -m dfp app corpus word_docs/ --name word
python -m dfp train --corpus corpus            # saved as your 'default'
```

`dfp train` refuses to replace `default` with a model that knows fewer
profiles than this one (use `--save-as NAME`, or `--force`).

A model records the feature set it was trained with. If the feature code
changes, loading it fails with a message to retrain; regenerate this file with
`scripts/reproduce.sh` (see `results/README.md`).
