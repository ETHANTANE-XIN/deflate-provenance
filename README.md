# DeflateProvenance (`dfp`)

**A digital forensics tool that reads the DEFLATE bitstream inside ZIP, GZIP,
DOCX, XLSX, PPTX, APK and JAR files and estimates which known compressor
profile produced it.** It reports a calibrated confidence or "unknown
encoder", checks whether an archive's entries agree with each other and with
the producer the file claims, and explains which features drove each decision.

ICT3215 Digital Forensics project. The design follows the team's project
proposal; the table in section 2 maps every proposal requirement to the code
that implements it.

---

## 1. What it does

A **profile** is a compressor implementation, version and setting, such as
"zlib 1.3 at level 6". DEFLATE (RFC 1951) lets each compressor choose its own
matches, block boundaries and Huffman codes, so different implementations
compress the same content into different bytes. `dfp` parses those bytes with
its own bit-level decoder, extracts statistical and *decision* features, and
compares them with reference profiles built from files it compressed itself.

For an archive it then asks the examiner's questions:

* Which known profile do the compressed streams match, and how confidently?
* Is the stream too small to say ("insufficient evidence"), or unlike every
  known profile ("unknown encoder")?
* Do all entries come from the same encoder, or was the archive rebuilt or
  partly edited?
* Does the stream evidence agree with the producer the container claims
  (`docProps/app.xml`, ODF `meta.xml`, a JAR manifest) and with the ZIP
  compression-option bits that producer writes?

The motivating case from the proposal works end to end: a DOCX whose
`docProps/app.xml` names Microsoft Word, but whose parts are exact zlib
output, is flagged three ways (profile excluded for Word, option bits
"normal" instead of Word's "super fast", entries exactly reproduced by zlib).
python-docx really produces such files: it writes "Microsoft Macintosh Word"
while compressing with zlib.

The report states what the tool does **not** claim: it does not prove
tampering or authorship, and because DEFLATE is lossless it cannot see an
encoder that was used before the last recompression. It reports whether the
current compressed data is consistent with the claimed origin.

---

## 2. Proposal alignment

| Proposal requirement | Where it is implemented |
|---|---|
| Accept ZIP, GZIP, DOCX, XLSX, PPTX, APK, JAR; locate every stream | `dfp/containers.py` |
| Own bit-level DEFLATE parser, not zlib | `dfp/deflate.py`, `dfp/bitreader.py` (zlib is only a test oracle) |
| Record block types and boundaries, code lengths and their header encoding, every literal and match, final padding | `dfp/deflate.py` (`BlockRecord`, `StreamRecord`) |
| Statistical features | `dfp/features.py` (182 features) |
| Decision features: longest match taken, lazy deferral, block boundaries, distance from the optimal Huffman table | `dfp/decisions.py` (27 features), `dfp/huffman.py` (exact package-merge) |
| Random Forest; per-prediction feature contributions | `dfp/ml/forest.py` (decision-path contributions that sum exactly to the prediction) |
| Confidence calibrated on held-out data | `dfp/ml/classifier.py`: a fifth of the training *source files* is held out; temperature fitted there |
| "Unknown encoder" when confidence is below a threshold or the sample is farther than a set distance from every profile | `dfp/ml/classifier.py`; both thresholds are set on the held-out source files |
| Programs sharing a library share a profile (Java on zlib) | `dfp/corpus.py` verifies byte-identical output before merging; rates are in the manifest |
| Identical outputs labelled with the whole set; any member counts as correct | `dfp/corpus.py` (label sets), `dfp/evaluate.py` (set-aware scoring) |
| Archive vote weighted by compressed size; stored or small entries are "insufficient evidence" | `dfp/aggregate.py` (minimum size measured on held-out data) |
| Flag a profile that contradicts the claimed producer or its ZIP option bits | `dfp/producers.py` (producer table), `dfp/aggregate.py` |
| Flag entries made by different encoders | `dfp/aggregate.py` (names the odd entries) |
| Report: profile, confidence, main features, inconsistencies | `dfp/report.py` (HTML + JSON) |
| Five components | (1) `containers.py`, `deflate.py`; (2) `features.py`, `decisions.py`; (3) `ml/`, `aggregate.py`, `producers.py`, `report.py`; (4) `corpus.py`, `encoders/`, `realfiles.py`; (5) `baselines.py`, `zipwriter.py`, `adversarial.py`, `evaluate.py` |
| Profiles: zlib 1-9 (Python and Java), zlib-ng (.NET 9+), Chromium's zlib (Node.js), libdeflate, zopfli, 7-Zip, Go | `dfp/encoders/` (plus .NET 8 and libarchive, both verified to share the zlib profile) |
| Exact versions recorded; pinned container image | corpus `manifest.json`; `requirements.txt`; `Dockerfile` |
| Applications such as Word profiled from saved documents, half kept for testing; producer table | `dfp app` (`dfp/realfiles.py`), `dfp/producers.py` |
| Split by source file | `dfp/evaluate.py` (`split_sources`; the report prints the overlap check) |
| Second test set from a different corpus (e.g. Govdocs1) and real application files | `dfp evaluate --second-sources`, `--real`, `--libreoffice`, `--python-docx`; `scripts/fetch_govdocs1.py` |
| Accuracy, precision, recall, macro-F1, confusion matrices per profile and size band; accuracy against compressed size | `dfp/evaluate.py`, `dfp/report.py` |
| Whole encoders left out of training (rejection vs misattribution) | `dfp/evaluate.py` (`leave_one_encoder_out`) |
| False-alarm rate; one part edited by a Python script | `dfp/realfiles.py`, `dfp/evaluate.py` |
| Baselines: preflate-rs estimate, brute-force zlib re-encoding, container metadata alone | `dfp/baselines.py`, `tools/preflate-estimate/` (preflate-rs 0.7.6) |
| Claimed producer rewritten to a false value | `dfp/zipwriter.py` (`impersonate`), `dfp/evaluate.py` |

---

## 3. Install

Python 3.10 or later (tested on 3.11) and the pinned bindings to the real C
encoders:

```
python -m pip install -r requirements.txt
```

External encoders are detected on `PATH` and used when present (the exact
versions behind the published results are in the `Dockerfile`): Java
(`javac`, `java`), Node.js, Go, .NET SDK, 7-Zip (`7z`), libarchive
(`bsdtar`), and LibreOffice (`soffice`) for real-file samples. Check what
this machine has with `python -m dfp encoders`.

The preflate-rs baseline is a small Rust wrapper (needs crates.io access):

```
cargo build --release --manifest-path tools/preflate-estimate/Cargo.toml
```

Or build the whole pinned environment: `docker build -t dfp .` (see the note
at the top of the `Dockerfile`).

---

## 4. Quick start

A trained model ships with the repository (`dfp/bundled_models/default.json.gz`,
7 profiles, see the notes beside it), so you can analyse files straight after
installing, with no training:

```bash
python -m dfp analyse suspicious.docx -o reports/   # uses the 'default' model
python -m dfp.demo                                  # the case studies, in seconds
python -m dfp models                                # which models are available
```

### Train once, reuse by name

Models are looked up by name: a file path if you give one, otherwise your
**local store** (`~/.dfp/models`, or `$DFP_MODELS`), then the **bundled**
models. `dfp train` saves into the local store, so a model you train yourself
(for example one that includes Word documents) replaces the bundled one
everywhere, and you only train again when the corpus changes.

```bash
# 1. Build the reference corpus once (every source x every encoder and setting)
python -m dfp corpus -o corpus --sources /usr/share/doc

# 2. (When available) add documents saved by an application, e.g. Word.
#    Half of the files train, half are held out for testing.
python -m dfp app corpus word_saved_docs/ --name word --description "Word 365, Windows"

# 3. Train and save as your 'default' (or --save-as NAME for another name)
python -m dfp train --corpus corpus

# 4. Use it: no -m needed for 'default'; -m NAME or -m FILE for others
python -m dfp analyse suspicious.docx -o reports/
python -m dfp analyse suspicious.docx -m word-2026 -o reports/

# The proposal's full evaluation (HTML + JSON report)
python -m dfp evaluate --corpus corpus -o reports/ \
    --second-sources govdocs1_sample/ --real real_docs/ --libreoffice 10 --python-docx 20
```

Other commands: `inspect` (dump every stream's structure), `reencode` (exact
re-encoding test against every panel encoder and setting), `producers` (the
producer table), `encoders` (versions), and the covert-channel extension
(`covert-embed`, `covert-detect`).

---

## 5. How it works

```
 file ─ containers.py ─ streams + container metadata + producer claims
          │
          ├─ deflate.py     bit-level RFC 1951 parser (blocks, trees, tokens, padding)
          ├─ features.py    182 statistical features
          ├─ decisions.py   27 decision features (choices versus alternatives)
          │
          ├─ ml/            Random Forest → profile + setting, calibrated confidence,
          │                 "unknown encoder", traced feature contributions
          │
          ├─ aggregate.py   entry statuses, vote weighted by compressed size,
          │                 mixed-encoder check, zlib re-encoding corroboration
          ├─ producers.py   claimed producer and option bits versus the evidence
          │
          └─ report.py      HTML + JSON report
```

**Decision features.** After decompression the tool knows the original bytes,
so it can compare each choice with the alternatives: whether the longest
match was taken (and the length at which the encoder stopped searching),
whether the nearest copy was chosen, whether a literal was followed by a
longer match (lazy matching) or a match was simply skipped, the minimum match
length, zlib's far-length-3 rule, how many symbols each block holds and
whether each split paid for its extra header, whether the cheapest block type
was used, and how far each Huffman table is from the optimal length-limited
table for the symbols the block really contains.

**Profiles.** The corpus builder compresses every source with every encoder
and setting, stores byte-identical outputs once with their whole label set,
and merges a program into its library's profile only after verifying
byte-identical output (Java, .NET 8 and libarchive all reproduce CPython's
zlib on 100% of streams, so they share the zlib profile, exactly as the
proposal predicted for Java).

**Thresholds are measured, not guessed.** A fifth of the training source files
is held out. On it the tool fits the calibration temperature, the confidence
threshold (the smallest one whose accepted predictions reach 95% precision),
the distance threshold (99th percentile of known-profile distances) and the
minimum compressed size from which every size band reaches 90% accuracy.

---

## 6. Evaluation results

Full report: [`results/evaluation.html`](results/evaluation.html) (raw numbers
in `results/evaluation.json`, corpus manifest with every version in
`results/manifest.json`, and the exact commands in `results/README.md`).
The training corpus has 348 source files (168 generated, 180 real files from
the reference machine), each compressed with every encoder and setting.

**Split by source file.** 243 training and 105 test source files (5,890 and
2,673 streams); **0** test streams share a source file with training. 92 test
streams are produced identically by more than one profile and are scored
against their whole label set.

**Closed set (test sources).** Accuracy **91.6%**, macro-F1 **0.913**;
the model answers 90.1% of streams and is right on **95.0%** of those.

| Profile | Test streams | Precision | Recall | F1 |
|---|---|---|---|---|
| 7-Zip | 307 | 0.93 | 0.90 | 0.91 |
| Chromium zlib (Node.js) | 368 | 0.86 | 0.92 | 0.89 |
| Go compress/flate | 379 | 1.00 | 1.00 | 1.00 |
| libdeflate | 384 | 0.86 | 0.82 | 0.84 |
| zlib (Python, Java, .NET 8, libarchive) | 716 | 0.92 | 0.99 | 0.95 |
| zlib-ng | 345 | 0.90 | 0.79 | 0.84 |
| zopfli | 174 | 0.99 | 0.92 | 0.95 |

**Accuracy against compressed size.** Every band from **256 bytes** upwards
reaches at least 90% accuracy on the streams the model answers (0.91 at
256 B to 1 KiB, 0.97 to 1.00 between 4 and 64 KiB, 0.96 above 64 KiB). Below
256 bytes the model would answer only 56% of streams, so the analyser reports
entries that small as "insufficient evidence"; the 256-byte threshold was
measured on held-out training sources, not chosen by hand.

**Setting inference** (set-aware): zlib-ng 99%, libdeflate 94%, 7-Zip 93%,
Go 87%, Chromium zlib 82%, zlib 69% (zlib levels 4 to 9 often differ by only a
few bytes), zopfli 60% (5 versus 15 iterations).

**Unknown encoders.** With each profile left out of training in turn, the
share of its streams rejected as "unknown" ranges from 2% (libdeflate) to
41% (7-Zip); the rest are attributed to the closest relative (Chromium zlib
to zlib-ng, Go to Chromium zlib, zopfli to 7-Zip). The team's own encoder is
rejected 27% of the time. Open-set rejection of encoders that resemble a
known one is the weakest result and is reported as such.

**Baselines on the same 280 test streams** (40 per profile):

| Method | Correct | Wrong profile | No answer |
|---|---|---|---|
| DeflateProvenance | **83.6%** | 5.4% | 11.1% |
| preflate-rs 0.7.6 parameter estimate | 23.2% | 58.2% | 18.6% |
| Brute-force zlib re-encoding (81 settings) | 14.3% | 0% | 85.7% |

preflate-rs is right mainly on zlib and libdeflate and cannot represent Go,
7-Zip or zopfli; brute-force zlib can only ever answer "zlib".

**Archive experiments** (72 test archives written by six real ZIP writers:
Python zipfile, Java, .NET 8, libarchive, Go, 7-Zip):

* genuine archives: DeflateProvenance profile correct **100%**, container
  metadata baseline 83%, false alarms **0%**;
* container metadata rewritten to impersonate another writer: the
  metadata-only baseline followed the forgery **96%** of the time;
  DeflateProvenance kept its attribution **100%** and disagreed with the
  forged writer 100%;
* claimed producer rewritten to a false value: flagged **100%**, attribution
  kept 100%;
* one part edited and recompressed with a different library: flagged and the
  edited entry singled out **90%**; the misses are edited parts too small to
  attribute (100% when the part had enough evidence).

**Second test set** (120 Python, Perl and X11 files, a different corpus never
used in training): accuracy 87.1%, macro-F1 0.868, 93.1% on answered streams;
reliable from 256 bytes as well.

**Real application files** (30 documents saved by LibreOffice 24.2.7.2:
DOCX, ODT, XLSX): all 30 attributed to zlib, consistent with the measured
producer table; false alarms **0%**; claimed producer rewritten to Microsoft
Word (with Word's option bits copied): flagged **100%**; one part edited and
recompressed by Go: flagged 97%. An edit recompressed by Python's zlib at the
same level is **not** detectable (0%): it produces exactly the bytes
LibreOffice itself would write, so no stream analysis can see it. The
proposal's version of this test edits *Word* documents, whose encoder is not
zlib; see section 7.

**python-docx** (20 documents that claim "Microsoft Macintosh Word" but are
written by zlib): **20/20 flagged**.

---

## 7. What the team still has to supply

These parts of the proposal need material that could not be produced on the
machine that generated the results above; the code paths are implemented and
tested with stand-ins.

* **Microsoft Word documents.** Save a set of documents with Word, run
  `python -m dfp app corpus <dir> --name word`, retrain, and re-run
  `dfp evaluate --real <held-out Word files>`. The producer table already
  encodes the proposal's preliminary finding (zlib reproduces no Word part;
  Word sets "super fast"); once Word files are added its expected profile is
  confirmed rather than inferred. Excel and PowerPoint entries are marked
  "not yet profiled" until the same is done for them.
* **Govdocs1.** Downloads were blocked in the build environment, so the
  second test set used a different local corpus (Python, Perl and X11 files)
  instead. Run `python scripts/fetch_govdocs1.py --threads 0 1 --out gd` and
  `dfp evaluate --second-sources gd` to use Govdocs1 as proposed.
* **Time-permitting items** (Google Docs exports, APKs built with the Android
  build tools): add them with `dfp app` in the same way.
* **Container build.** The `Dockerfile` pins every version but was not
  test-built (no Docker daemon was available); build it once and commit the
  resulting digest.

---

## 8. Limitations

* A profile names the compression library, not the program: every zlib-based
  program (Python, Java, .NET 8, libarchive, LibreOffice) shares one profile,
  and only container metadata separates them.
* Small streams carry little evidence; below the measured minimum size the
  tool reports "insufficient evidence" instead of guessing.
* Rejecting encoders never seen in training is hard when they resemble a known
  one (see the unknown-encoder results); an unseen encoder is often attributed
  to its closest relative.
* Real content differs from generated content; the second test set measures
  how much that costs.

---

## 9. Extensions beyond the proposal

These were in the earlier prototype and are kept, clearly separated:

* PDF (`/FlateDecode`), PNG, EPUB, ODF, raw zlib and headerless DEFLATE input;
* deterministic signature rules (`dfp/signatures.py`);
* a DEFLATE padding covert channel and its detector (`dfp/covert.py`,
  `covert-embed`, `covert-detect`).

---

## 10. Tests

```
python -m unittest discover -s tests
```

54 tests cover the bit reader, the parser against zlib (350 streams across all
levels and strategies, plus truncation), optimal Huffman construction, decision
features, every available encoder, the corpus (unique sources, label sets,
Java sharing the zlib profile, regeneration), calibration and explanations,
the vote, producer claims and the producer table, ZIP rewriting, the
baselines, verification by re-encoding, metadata robustness, signatures, the
covert-channel extension and the size-band reliability rule. Tests that need
an optional encoder or the preflate-rs tool skip when it is missing.

---

## 11. Repository layout

```
deflate-provenance/
  dfp/
    containers.py   ZIP/OOXML/APK/JAR/ODF/EPUB/GZIP/zlib/PNG/PDF/raw + claims
    bitreader.py    LSB-first bit reader
    deflate.py      bit-level RFC 1951 parser (table-driven, no zlib)
    features.py     statistical features
    decisions.py    decision features
    huffman.py      optimal and length-limited Huffman lengths
    encoders/       zlib, java, zlib-ng, node, libdeflate, zopfli, 7zip, go,
                    dotnet, libarchive, purepy
    corpus.py       reference corpus, label sets, profile sharing, manifest
    training.py     train a model from a corpus
    modelstore.py   find models by name (local store, then bundled)
    bundled_models/ the trained 'default' model shipped with the repository
    ml/             forest.py, classifier.py
    aggregate.py    archive analysis and consistency findings
    producers.py    producer table
    report.py       HTML + JSON reports
    baselines.py    preflate-rs, brute-force zlib, metadata-only
    zipwriter.py    lossless ZIP rewriting
    realfiles.py    application files, LibreOffice samples, real-file tests
    evaluate.py     evaluation plan
    adversarial.py  metadata robustness, exact re-encoding
    signatures.py   deterministic rules (extension)
    covert.py       covert channel (extension)
    charts.py       SVG charts
    cli.py, demo.py
  tools/preflate-estimate/   Rust wrapper around preflate-rs 0.7.6
  scripts/fetch_govdocs1.py  Govdocs1 sample downloader
  results/                   the published evaluation (HTML, JSON, manifest)
  tests/test_dfp.py
  Dockerfile, requirements.txt
```

---

## 12. Use of generative AI

Per the assignment's disclosure requirement, record in your submission where
AI assistance was used (for example drafting code and documentation). The
correctness and academic integrity of the final work are the team's
responsibility: validate every result independently (the test suite and the
`evaluate` and `demo` outputs are the primary validation here).
