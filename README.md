# dfp — DEFLATE Compression-Provenance Toolkit

**A digital-forensics tool that attributes a compressed file to the encoder
implementation that produced it, by reading the raw DEFLATE bitstream — not the
container metadata.**

ICT3215 Digital Forensics project. Side chosen: **defensive** (a forensic
attribution tool) **plus** a responsible anti-forensics component (an
output-preserving covert channel and its detector).

---

## 1. What it does and why it matters

Files as different as ZIP archives, Word `.docx`, Android `.apk`, Java `.jar`,
EPUB, **PDF**, PNG and GZIP all compress their contents with the same algorithm,
**DEFLATE** (RFC 1951). DEFLATE leaves the *encoder* many free choices — block
types and split points, Huffman tree shapes, the code-length alphabet, greedy
vs lazy matching, search depth. Different implementations (zlib, libdeflate,
Java, .NET, Go, Info-ZIP, 7-Zip, …) make **different** choices for the *same*
input, and none of those choices is recorded anywhere in the file.

Existing provenance work fingerprints the **container metadata** — timestamps,
filenames, ZIP headers, host-system bytes. All of that is editable in seconds.
`dfp` fingerprints the **compressed bitstream itself**, which an adversary
cannot forge without re-running the original encoder byte-for-byte.

Forensic uses:

* a `.docx` that claims to be Microsoft Word but whose streams were compressed
  by a different toolchain → a **rebuilt / backdated document**;
* an `.apk` rebuilt by a repackaging tool rather than the original build →
  **repackaging**;
* an evidence archive silently recompressed after seizure → a **chain-of-custody
  break**.

---

## 2. Install

Pure Python, CPU-only, standard library **plus NumPy**. No scikit-learn, no
matplotlib, no native build.

```
python -m pip install numpy
```

Optional external encoders enrich the training corpus if present on `PATH`
(the tool auto-detects them and simply omits any that are missing):
**Java** (`javac`/`java`), **.NET** (`dotnet`), **Node.js** (`node`),
**libarchive** (`bsdtar`/`tar`). CPython's `zlib` and the built-in pure-Python
encoder are always available.

Run from the project root (the folder containing this file):

```
python -m dfp --help
```

---

## 3. Quick start

```bash
# 1. Train a model (builds a corpus by compressing generated files with every
#    available encoder, then trains a random forest). Default scheme = coarse.
python -m dfp train -o model.json

# 2. Attribute a file and write a court-ready report
python -m dfp analyse suspicious.docx -m model.json -o reports/

# 3. Reproduce all evaluation figures (confusion matrix, minimum-evidence
#    curve, open-set, feature importance) as a self-contained HTML report
python -m dfp evaluate -o reports/

# 4. Run the case studies (forged document, repackaging, metadata robustness,
#    covert channel)
python -m dfp.demo
```

---

## 4. Supported input formats

Verified by `tests/` and by direct probing. **DEFLATE is what the tool reads**, so
a file works if its compression method is DEFLATE.

| Input | Detected as | Works |
|---|---|---|
| `.zip` | `zip` | ✅ |
| `.docx` `.xlsx` `.pptx` `.docm` | `ooxml` | ✅ |
| `.apk` | `apk` | ✅ |
| `.jar` `.aar` | `jar` | ✅ |
| `.epub` | `epub` | ✅ |
| `.odt` `.ods` `.odp` | `odf` | ✅ |
| `.pdf` (`/FlateDecode` streams) | `pdf` | ✅ |
| `.gz` `.tgz` | `gzip` | ✅ |
| zlib stream (`.zz`) | `zlib` | ✅ |
| `.png` (IDAT) | `png` | ✅ |
| headerless raw DEFLATE | `raw` | ✅ |

**Not DEFLATE, so out of scope** (the tool reports them rather than failing):

* ZIP entries stored uncompressed (method 0) → counted as `stored`, no bitstream
  to analyse.
* ZIP entries using BZIP2 / LZMA / XZ / Zstandard / PPMd (methods 12, 14, 93–98)
  → counted as `other_method`.
* `.7z`, `.rar`, `.xz`, `.bz2`, `.zst`, `.br` archives — different algorithms
  entirely.
* Encrypted archives and encrypted PDFs — the compressed bytes are not readable.
* PDF streams behind a filter chain (e.g. `/Filter [/ASCII85Decode /FlateDecode]`)
  → skipped, because the on-disk bytes are not raw DEFLATE.

---

## 5. Command reference

| Command | Purpose |
|---|---|
| `dfp inspect FILE` | Parse and dump every DEFLATE stream: block types, sizes, token counts, padding. Uses **no zlib** for parsing. |
| `dfp train [-o M] [--scheme S] [--trees N] [--per-combo K]` | Build a corpus and train the classifier. `--scheme` ∈ `fine`/`lineage`/`coarse` (see §6). |
| `dfp analyse FILE -m M [-o DIR]` | Attribute a file; writes `<name>.report.html` and `.json`. |
| `dfp evaluate [-o DIR] [--scheme S]` | Full quantitative evaluation + figures. |
| `dfp recompress FILE` | Constructive test: try to reproduce each stream by recompressing its output with every panel encoder. An exact byte match names the encoder+level. |
| `dfp covert-embed FILE MSG -o OUT` | Hide `MSG` in DEFLATE final-byte padding; the decompressed bytes are unchanged. |
| `dfp covert-detect FILE` | Flag streams whose padding is non-zero (a padding covert channel). |

---

## 6. How it works (pipeline)

```
 input file
    │  containers.py   locate raw DEFLATE streams + harvest container metadata
    ▼
 raw DEFLATE stream
    │  deflate.py      bit-level RFC 1951 parser (records every encoder choice)
    ▼
 StreamRecord (blocks, trees, tokens, padding)
    │  features.py     182-dim encoder-behaviour vector
    ▼
 feature vector
    │  ml/             random forest + temperature calibration + abstain
    │  signatures.py   deterministic hard rules (transparent corroboration)
    ▼
 per-stream verdict
    │  aggregate.py    size/confidence-weighted vote per archive
    ▼
 report.py            court-ready HTML + JSON
```

Key modules:

* **`bitreader.py`** — LSB-first bit reader tracking absolute bit position (so
  block boundaries, an encoder artefact, are recorded exactly).
* **`deflate.py`** — the forensic parser. Decodes stored/static/dynamic blocks
  from first principles and keeps block boundaries, the 19-symbol code-length
  alphabet and its repeat usage, both Huffman trees as code-length arrays, the
  token sequence with order statistics (the lazy-match signature), and the
  final padding bits. Verified byte-for-byte against `zlib` on 350+ streams
  (all 10 levels × 5 strategies × 7 content types).
* **`encoders/`** — adapters for zlib (10 levels × 5 strategies), a **from-scratch
  pure-Python DEFLATE encoder** (greedy & lazy; also the covert-channel host and
  the open-set "unknown"), Java, .NET, Node.js and libarchive.
* **`ml/`** — a pure-NumPy random forest (weighted-Gini CART, bootstrap, random
  feature subsets), temperature-calibrated on out-of-bag probabilities, with an
  **abstain** option driven by confidence, top-2 margin and a novelty score.
  Serialises to plain JSON (no pickle).
* **`covert.py`** — the covert channel (in encoder freedom) and its detector.
* **`adversarial.py`** — metadata normalisation, recompression detection, and
  the metadata-vs-bitstream comparison.

---

## 7. Attribution granularity (`--scheme`)

Vendor names are **not always separable**, because several tools link the same
zlib code base and emit near-identical bytes. Rather than pretend otherwise, the
tool lets you choose how specific a claim the evidence supports:

* **`fine`** — every family separate (`zlib`, `java`, `dotnet`, `node`,
  `libarchive`, `purepy`). Highest resolution, lowest confidence where families
  overlap.
* **`lineage`** — merge the indistinguishable zlib-code-base tools
  (`libarchive`→`zlib`), keep the rest separate. Evaluation default.
* **`coarse`** — `zlib_lineage` (the whole zlib toolchain, incl. Java, Node,
  .NET's managed port) vs `non_standard` (a custom encoder; our `purepy` is the
  exemplar). Highest-confidence, court-usable claim. **Training default.**

This design choice is itself a finding: see §8.

---

## 8. Evaluation results

Generated by `python -m dfp evaluate` (numbers vary a little with `--per-combo`).
Representative run:

* **zlib strategy inference:** ~88 % accuracy (default / filtered / huffman-only
  / RLE / fixed).
* **zlib compression-level band inference:** ~86 % accuracy (L0 / L1-3 / L4-6 /
  L7-9).
* **Minimum-evidence curve:** accuracy-on-answered rises from ~0.84 at 4 KB to
  ~0.92 at 64–128 KB; the abstain option withholds a verdict below ~2 KB (too
  little evidence). "Attribution is reliable from N bytes" is the headline.
* **Open-set:** an encoder never seen in training is rejected rather than
  misattributed (rejection rate reported per run).
* **Adversarial (metadata robustness):** normalising every ZIP metadata field
  changes the metadata but moves the bitstream feature vector by **0.0** — the
  invariance the whole approach rests on.
* **Recompression:** exact byte-for-byte reproduction identifies the encoder +
  level constructively (strongest possible attribution).
* **Covert channel:** a message is embedded in DEFLATE padding with the
  decompressed output provably unchanged; the corpus detector's non-zero-padding
  fraction jumps from ~0.00 (clean) to ~0.46 (stego).

Figures (confusion matrix, curves, importances) are rendered as inline SVG in
`reports/evaluation.html`.

---

## 9. Honest limitations

* **Same-code-base tools are not separable by vendor name.** zlib, Info-ZIP,
  libarchive, Java's `Deflater`, Node's zlib and .NET's managed zlib port emit
  near-identical bytes. The tool groups them (`coarse`/`lineage`) and abstains
  rather than guessing a vendor the bytes cannot support.
* **A library, not the program.** An attribution names the *compression
  implementation*, not the application that called it (many programs link
  zlib).
* **Small streams carry little evidence.** Below ~2 KB the tool usually
  abstains; this is by design (see the minimum-evidence curve).
* **Optimal convergence.** On highly repetitive input, competent encoders can
  converge on the *same* optimal encoding, making provenance genuinely
  undecidable there.

Every report states: **the attribution is supporting evidence, not proof of
authorship.**

---

## 10. Tests

```
python -m unittest discover -s tests
```

24 tests cover the bit reader, the parser (round-trip vs zlib across all
levels/strategies), every encoder, container extraction (ZIP/GZIP with CRC
verification), feature stability, the covert channel, metadata-robustness
invariance, recompression, the signature rules, and classifier train/predict/
save/load.

---

## 11. Repository layout

```
deflate-provenance/
  dfp/
    bitreader.py      LSB-first bit reader
    deflate.py        bit-level RFC 1951 parser (no zlib for parsing)
    containers.py     ZIP/OOXML/APK/JAR/EPUB/GZIP/zlib/PNG/raw + metadata
    features.py       182-feature encoder-behaviour vector
    encoders/         zlib, purepy, java, dotnet, node, libarchive adapters
    ml/               forest.py, classifier.py (pure-NumPy, calibrated, abstain)
    schemes.py        fine / lineage / coarse label granularity
    signatures.py     deterministic hard rules
    aggregate.py      per-stream analysis + per-archive vote
    adversarial.py    metadata normalisation, recompression, channel compare
    covert.py         covert channel + detector
    evaluate.py       closed-set / minimum-evidence / open-set / adversarial
    charts.py         dependency-free SVG figures
    report.py         court-ready HTML + JSON
    cli.py            command-line interface
    demo.py           case studies
  tests/
    test_dfp.py       unittest suite
  README.md
  requirements.txt
```

---

## 12. Use of generative AI

Per the assignment's disclosure requirement, record in your submission where AI
assistance was used (e.g. drafting code and documentation), and remember that
the correctness and academic integrity of the final work are the team's
responsibility — validate every result independently (the test suite and the
`evaluate`/`demo` outputs are the primary validation here).
