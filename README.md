# DeflateProvenance (`dfp`)

**A digital forensics tool that reads the DEFLATE bitstream inside ZIP, GZIP,
DOCX, XLSX, PPTX, APK and JAR files and estimates which known compressor
profile produced it.** It reports a calibrated confidence or "unknown
encoder", checks whether an archive's entries agree with each other and with
the producer the file claims, and explains which features drove each decision.

ICT3215 Digital Forensics project, version 2.1.0. The design follows the
team's project proposal; the table in section 2 maps every proposal
requirement to the code that implements it. `CHANGELOG.md` lists what changed
since 2.0.0.

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
* Do all entries come from the same writer, or was the archive rebuilt or
  partly edited?
* Does the evidence agree with the producer the container claims
  (`docProps/app.xml`, ODF `meta.xml`, a JAR manifest) and with the ZIP
  compression-option bits that producer writes?

Conclusions of the last two kinds rest on **exact** evidence only, never on
the statistical model alone: an entry that a reference encoder reproduces byte
for byte, and the way each stream *ends*. A writer ends every DEFLATE stream
the same way (zlib's finish leaves a final block that carries data, Microsoft
Office writes a sync flush and an empty final block, Go writes an empty stored
final block), so entries that end differently were written by different
programs. When the statistical model and the exact evidence disagree, or the
model alone sees two encoders, the report says "cannot confirm".

The motivating case from the proposal works end to end: a DOCX whose
`docProps/app.xml` names Microsoft Word, but whose parts were written by
zlib's finish, is flagged twice (its parts do not end the way Word ends every
part, and its option bits are "normal" instead of Word's "super fast").
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
| Accept ZIP, GZIP, DOCX, XLSX, PPTX, APK, JAR; locate every stream | `dfp/containers.py` (also ZIP64, self-extracting stubs, multi-member GZIP, encrypted entries listed but never parsed) |
| Own bit-level DEFLATE parser, not zlib | `dfp/deflate.py`, `dfp/bitreader.py` (zlib is only a test oracle) |
| Record block types and boundaries, code lengths and their header encoding, every literal and match, final padding | `dfp/deflate.py` (`BlockRecord`, `StreamRecord`) |
| Statistical features | `dfp/features.py` (186 features) |
| Decision features: longest match taken, lazy deferral, block boundaries, distance from the optimal Huffman table | `dfp/decisions.py` (27 features), `dfp/huffman.py` (exact package-merge) |
| Random Forest; per-prediction feature contributions | `dfp/ml/forest.py` (decision-path contributions that sum exactly to the prediction) |
| Confidence calibrated on held-out data | `dfp/ml/classifier.py`: a fifth of the training *source files* is held out; temperature fitted there |
| "Unknown encoder" when confidence is below a threshold or the sample is farther than a set distance from every profile | `dfp/ml/classifier.py`: confidence threshold at 95% held-out precision; distance is a nearest-neighbour novelty score on the decision features, thresholded at the 99th percentile of known streams |
| Programs sharing a library share a profile (Java on zlib) | `dfp/corpus.py` verifies byte-identical output before merging; rates are in the manifest |
| Identical outputs labelled with the whole set; any member counts as correct | `dfp/corpus.py` (label sets), `dfp/evaluate.py` (set-aware scoring) |
| Archive vote weighted by compressed size; stored or small entries are "insufficient evidence" | `dfp/aggregate.py` (minimum size: held-out measurement, never below 256 bytes) |
| Flag a profile that contradicts the claimed producer or its ZIP option bits | `dfp/producers.py` (producer table with measured basis), `dfp/aggregate.py` |
| Flag entries made by different encoders | `dfp/aggregate.py` (stream endings and exact proofs; names the odd entries) |
| Report: profile, confidence, main features, inconsistencies | `dfp/report.py` (HTML + JSON) |
| Five components | (1) `containers.py`, `deflate.py`; (2) `features.py`, `decisions.py`; (3) `ml/`, `aggregate.py`, `producers.py`, `report.py`; (4) `corpus.py`, `encoders/`, `realfiles.py`; (5) `baselines.py`, `zipwriter.py`, `adversarial.py`, `evaluate.py` |
| Profiles: zlib 1-9 (Python and Java), zlib-ng (.NET 9+), Chromium's zlib (Node.js), libdeflate, zopfli, 7-Zip, Go | `dfp/encoders/` (plus .NET 8 and libarchive, both verified to share the zlib profile) |
| Exact versions recorded; pinned container image | corpus `manifest.json`; `requirements.txt`; `Dockerfile` (base image by digest, Ubuntu snapshot, checksummed downloads) |
| Applications such as Word profiled from saved documents, half kept for testing; producer table | `dfp app` (`dfp/realfiles.py`), `dfp/producers.py`; the shipped model includes a Word profile |
| Split by source file | `dfp/evaluate.py` (split by source *family*; independent overlap checks in the report) |
| Second test set from a different corpus (e.g. Govdocs1) and real application files | `dfp evaluate --second-sources`, `--real`, `--libreoffice`, `--python-docx`; `scripts/fetch_govdocs1.py` |
| Accuracy, precision, recall, macro-F1, confusion matrices per profile and size band; accuracy against compressed size | `dfp/evaluate.py`, `dfp/report.py` |
| Whole encoders left out of training (rejection vs misattribution) | `dfp/evaluate.py` (`leave_one_encoder_out`) |
| False-alarm rate; one part edited by a Python script | `dfp/realfiles.py`, `dfp/evaluate.py` |
| Baselines: preflate-rs estimate, brute-force zlib re-encoding, container metadata alone | `dfp/baselines.py`, `tools/preflate-estimate/` (preflate-rs 0.7.6) |
| Claimed producer rewritten to a false value | `dfp/zipwriter.py` (`impersonate`), `dfp/evaluate.py` |

---

## 3. Install

Python 3.10 or later (the published results used Python 3.12 in the reference
container; the tool also runs on Windows with Python 3.13) and the pinned
bindings to the real C encoders:

```bash
python -m pip install -r requirements.txt
```

External encoders are detected on `PATH` and used when present (the exact
versions behind the published results are in the `Dockerfile`): Java
(`javac`, `java`), Node.js, Go, .NET SDK, 7-Zip (`7z`), libarchive
(`bsdtar`), and LibreOffice (`soffice`) for real-file samples. On Windows,
7-Zip is also found in its install folder and Windows' own `tar.exe` (bsdtar)
is used as libarchive. Check what this machine has with
`python -m dfp encoders`. Exact verification uses whichever reference
encoders are installed, and every report lists the ones that were missing.

The preflate-rs baseline is a small Rust wrapper (needs crates.io access):

```bash
cargo build --release --manifest-path tools/preflate-estimate/Cargo.toml
```

Or build the whole pinned environment, which is how the published results
were produced (see the note at the top of the `Dockerfile`):

```bash
docker build -t dfp .
```

---

## 4. Quick start

A trained model ships with the repository (`dfp/bundled_models/default.json.gz`,
8 profiles including Microsoft Word, see the notes beside it), so you can
analyse files straight after installing, with no training:

```bash
python -m dfp analyse suspicious.docx -o reports/
```

```bash
python -m dfp.demo
```

```bash
python -m dfp models
```

The first writes an HTML and a JSON report; the second runs the case studies
in seconds; the third lists the available models.

### Train once, reuse by name

Models are looked up by name: a file path if you give one, otherwise your
**local store** (`~/.dfp/models`, or `$DFP_MODELS`), then the **bundled**
models. `dfp train` saves into the local store, so a model you train yourself
replaces the bundled one everywhere, and you only train again when the corpus
changes. It refuses to replace `default` with a model that knows fewer
profiles than the bundled one (use `--save-as NAME` or `--force`).

```bash
# 1. Build the reference corpus once (every source x every encoder and setting)
python -m dfp corpus -o corpus --sources /usr/share/doc

# 2. Add documents saved by an application, e.g. Word.
#    Half of the files train, half are held out for testing (the command lists them).
python -m dfp app corpus word_saved_docs/ --name word --description "Word 365, Windows"

# 3. Train and save as your 'default' (or --save-as NAME for another name)
python -m dfp train --corpus corpus

# 4. Use it: no -m needed for 'default'; -m NAME or -m FILE for others
python -m dfp analyse suspicious.docx -o reports/
```

The whole published evaluation, corpus and bundled model are regenerated by one
command inside the reference image (see `results/README.md`):

```bash
docker run --rm -e DFP_WORKERS=16 -v "$PWD/out:/out" -v "$PWD/word:/word:ro" --entrypoint sh dfp /opt/dfp/scripts/reproduce.sh /out /word
```

Other commands: `inspect` (dump every stream's structure), `reencode` (exact
re-encoding test against every panel encoder and setting), `producers` (the
producer table), `encoders` (versions), `evaluate` (the evaluation on your own
corpus), and the covert-channel extension (`covert-embed`, `covert-detect`).

---

## 5. How it works

```
 file ─ containers.py ─ streams + container metadata + producer claims
          │
          ├─ deflate.py     bit-level RFC 1951 parser (blocks, trees, tokens, padding)
          ├─ features.py    186 statistical features
          ├─ decisions.py   27 decision features (choices versus alternatives)
          │
          ├─ ml/            Random Forest → profile + setting, calibrated confidence,
          │                 "unknown encoder" (confidence and novelty), feature contributions
          │
          ├─ aggregate.py   entry statuses, exact re-encoding proofs, vote weighted by
          │                 compressed size, stream endings, mixed-encoder check
          ├─ producers.py   claimed producer, its stream endings and option bits versus the evidence
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
table for the symbols the block really contains. Flush markers and the
end-of-stream terminator have features of their own, so a writer's flushing
habit is not mistaken for its block-splitting strategy.

**Profiles.** The corpus builder compresses every source with every encoder
and setting, stores byte-identical outputs once with their whole label set,
and merges a program into its library's profile only after verifying
byte-identical output (Java, .NET 8 and libarchive all reproduce CPython's
zlib on 100% of streams, so they share the zlib profile, exactly as the
proposal predicted for Java). Microsoft Word, which has no reference encoder,
is an *application profile* learnt from documents saved by Word.

**Thresholds are measured, not guessed.** A fifth of the training source files
is held out. On it the tool fits the calibration temperature, the confidence
threshold (the smallest one whose accepted predictions reach 95% precision), a
second, high-precision threshold (99%), the novelty threshold (99th
percentile of known profiles' nearest-neighbour distances on the decision
features) and the minimum compressed size from which held-out attribution is
reliable (never below 256 bytes).

**Archive verdict.** Each entry is attributed, "unknown encoder" or
"insufficient evidence". Entries are then re-encoded exactly: every entry with
CPython zlib at all 81 level/memLevel settings (with and without a sync
flush), and doubtful entries with the reference encoders of the leading
profile, of the model's most probable profiles and of encoders proven
elsewhere in the archive. A byte-for-byte match counts as **proof** only on an
entry of at least 1 KiB with at least 64 LZ77 matches: below that, encoders
coincide (every encoder writes incompressible data as the same stored blocks,
and small Word parts match zlib). The vote is weighted by compressed size; a
profile is reported only when entries matching no known profile are not the
majority, otherwise the verdict is "unknown" with the closest profile shown.

**Mixed encoders** are reported as inconsistent in ZIP, OOXML, ODF, JAR and
EPUB files on exact evidence only: entries whose streams end differently
(streams with almost no LZ77 matches, such as small images, are left out), or
entries proven by reference encoders of two different families with no
encoder reproducing both. Libraries behind the zlib API (zlib, zlib-ng,
Chromium's zlib) count as one family, because they coincide with zlib on some
inputs. The entries named are those of the smaller side, since an edit touches
few parts; a tie names none. An APK is assembled by copying compressed entries
from libraries unchanged, and streams in PDFs, PNGs and GZIP files come from
different writers routinely, so there it is information only.

**Producer claims** are checked against the producer table
(`python -m dfp producers`), whose every entry records its basis: the profiles
the producer writes, how it ends its streams, and the ZIP option bits it sets.
Stream endings and option bits are checked exactly; a contradiction from the
model needs an established verdict (proven, or every entry above the 99%
precision threshold). Some Word builds compress parts exactly as zlib level 1
with a sync flush does, so a zlib match is no evidence against Word.

---

## 6. Evaluation results

Full report: [`results/evaluation.html`](results/evaluation.html) (raw numbers
in `results/evaluation.json`, corpus manifest with every version in
`results/manifest.json`, and the exact commands in `results/README.md`). Every
number below was produced by `scripts/reproduce.sh` in the reference image.

**Corpus.** 286 source files (168 generated, 118 real files from the image),
each compressed with every encoder and setting: 9,146 streams. Java, .NET 8
and libarchive reproduced CPython's zlib on 100% of their streams and share
the zlib profile. The Word profile comes from 60 documents saved by Word 16
(674 streams): 30 train and 30 are held out. Documents saved from one template
share byte-identical parts, so 208 held-out streams that also occur in a
training document were moved to training, leaving 129 distinct Word test
streams.

**Split by source family.** Files from one directory that share a name
prefix, or whose content is a near-duplicate, form one family and stay on one
side: 201 training and 85 test source files (6,353 training and 2,689 test
streams, including the 129 Word streams). The independent checks find **0**
test sources identical or near-identical to a training source and **0**
held-out Word streams identical to a training stream. 131 test streams are
produced identically by more than one profile and are scored against their
whole label set.

**Closed set (test sources).** End to end, scoring what `dfp analyse`
reports for each stream ("unknown encoder" and "insufficient evidence" count
as misses): accuracy **81.2%**, macro-F1 **0.877**. The tool answers 84.5% of
streams and is right on **96.0%** of the streams it answers. With the unknown
rule and the minimum size switched off (forced choice), accuracy is 92.2% and
macro-F1 0.927.

| Profile | Test streams | Precision | Recall | F1 | F1, forced choice |
|---|---|---|---|---|---|
| 7-Zip | 246 | 0.95 | 0.76 | 0.84 | 0.90 |
| Chromium zlib (Node.js) | 290 | 0.93 | 0.67 | 0.78 | 0.84 |
| Go compress/flate | 305 | 1.00 | 0.83 | 0.91 | 1.00 |
| libdeflate | 299 | 0.93 | 0.77 | 0.84 | 0.88 |
| Microsoft Word 16 | 129 | 1.00 | 0.98 | 0.99 | 0.99 |
| zlib (Python, Java, .NET 8, libarchive) | 784 | 0.94 | 0.90 | 0.92 | 0.93 |
| zlib-ng | 494 | 0.98 | 0.76 | 0.86 | 0.91 |
| zopfli | 142 | 1.00 | 0.78 | 0.88 | 0.96 |

**Accuracy against compressed size.** Below 256 compressed bytes the tool
reports "insufficient evidence" (259 test streams; forced choice would be
right on 80% of them). Where the open-set rule lets the classifier answer, it
is right on 93% of streams from 256 bytes to 1 KiB, 98% from 1 to 4 KiB, 97%
to 99% between 4 and 64 KiB, and 95% above 64 KiB; end to end the same bands
score 82%, 94%, 97%, 94% and 84%. The held-out calibration fold measured no
minimum at all (every band met the 90% target), so the 256-byte floor applies.

**Setting inference** (set-aware): Word 100% (one setting), zlib-ng 96%,
7-Zip 95%, libdeflate 93%, Go 86%, Chromium zlib 84%, zlib 69% (zlib levels 4
to 9 often differ by only a few bytes), zopfli 64% (5 versus 15 iterations).

**Unknown encoders.** With each profile left out of training in turn, the
share of its streams rejected as "unknown" is 80% for zopfli, 32% for 7-Zip,
29% for zlib-ng, 26% for zlib, 24% for Go, 9% for libdeflate, 8% for Chromium
zlib and **0% for Word**: all 129 Word test streams are then attributed to
zlib. The rest go to the closest relative (Chromium zlib to zlib-ng, Go to
Chromium zlib). The team's own encoder, never trained on, is rejected 30% of
the time. The open-set rules reject 7.1% of held-out *known* streams.
Rejecting unseen encoders that resemble a known one is the weakest result.

**Baselines on the same 280 test streams** (40 per corpus profile):

| Method | Correct | Wrong profile | No answer | "Is it zlib?" precision / recall |
|---|---|---|---|---|
| DeflateProvenance | **83.2%** | 3.6% | 13.2% | 88.1% / 92.5% |
| preflate-rs 0.7.6 parameter estimate | 25.7% | 65.4% | 8.9% | 54.3% / 62.5% |
| Brute-force zlib re-encoding (81 settings) | 14.3% | 0% | 85.7% | 100% / 100% |

preflate-rs is right mainly on zlib and libdeflate and cannot represent Go,
7-Zip or zopfli. Brute-force zlib can only ever answer "zlib": it is a perfect
zlib detector, which is why `dfp analyse` uses it as proof, but not a profile
classifier.

**Archive experiments** (72 test archives, 12 from each of six real ZIP
writers: Python zipfile, Java, .NET 8, libarchive, Go and 7-Zip; each holds
five source files as DOCX parts plus `[Content_Types].xml` and
`docProps/app.xml`):

* genuine archives: DeflateProvenance profile correct **100%**; the container
  metadata baseline names the profile 100% of the time and the exact writer
  83%; false alarms **0%** (also 0% with reference encoders switched off);
* container metadata rewritten to impersonate another writer: the
  metadata-only baseline followed the forgery **96%** of the time;
  DeflateProvenance kept its attribution 100% and never agreed with the forged
  writer;
* claimed producer rewritten to a false value (with that producer's option
  bits): flagged **100%**, every time by the stream evidence; attribution kept
  100%;
* the largest part edited and recompressed by a library of another profile
  (Go for the zlib writers, zlib for Go and 7-Zip): flagged and the edited
  part singled out **97%** (70 of 72). The two misses are 7-Zip archives
  edited with zlib: both writers end their streams the same way, and exact
  re-encoding could prove only one side, so the report says "cannot confirm". With reference encoders switched
  off (CPython zlib only, as on a machine with nothing else installed) the
  rate is 83%: stream endings still catch every Go edit.

**Second test set** (120 Python, Perl and X11 files, a different corpus never
used in training): end to end 78.6% (macro-F1 0.824), answering 85.7% of
streams and right on 91.8% of those; forced choice 86.8%; reliable from 1 KiB.

**Real application files** (60 genuine files: the 30 held-out Word documents
and 30 documents saved by LibreOffice 24.2.7.2, 10 each of DOCX, ODT and
XLSX):

* false alarms **0%**; 58 are "consistent" (all 30 LibreOffice files as zlib,
  28 Word documents as Word) and 2 Word documents are "inconclusive" (the
  model leans to two profiles for their parts and nothing exact separates
  them);
* one part edited by a Python script (zlib) in each Word document: flagged and
  the edited part singled out **100%** (30/30). The 30 LibreOffice files are
  skipped for this test: zlib reproduces their parts exactly, so a zlib edit
  produces the very bytes LibreOffice writes, which no method can detect;
* one part edited by Go in every file: flagged and singled out **100%**
  (60/60);
* claimed producer rewritten (Word documents to LibreOffice, LibreOffice
  documents to Word, with that producer's option bits): flagged **100%**,
  every time by the stream evidence; attribution kept 100%.

**python-docx** (20 documents that claim "Microsoft Macintosh Word" but are
written by zlib): **20/20 flagged**.

**Files of unknown history.** As a field check, the final analyser was run
over the 2,072 ZIP-based files found on a team member's computer (2,039 of
them readable; only counts were recorded, and no file was kept). Their history
is unknown, so these are observations, not error rates:

* none of the 807 ZIP, JAR, EPUB, ODF and unclaimed OOXML files was reported
  inconsistent (APKs, where mixed encoders are normal, get information only);
* every one of the 51 files that claim Word but were written by a library
  (zlib's finish and "normal" option bits, such as python-docx's "Microsoft
  Macintosh Word") was flagged;
* 4 Word documents whose compressed parts are Word's but whose ZIP container
  was rewritten by another program ("version made by" 1.0 instead of Word's
  4.5, "normal" option bits) were flagged for their option bits;
* 7 Office files were reported as written by two programs: one Word document
  with several parts rewritten by a script, 3 documents last saved by Word for
  the web (which keeps parts written by desktop Word), and 3 PowerPoint or
  Excel files whose history is unknown.

---

## 7. What the team still has to supply

* **The Word documents behind the Word profile are not in the repository.**
  The 60 documents were generated from neutral text (Python documentation and
  generated prose) and saved by Word 16 on a team member's Windows PC; Word
  writes the account name into each document's `docProps/core.xml`, so they
  were kept out. To regenerate the Word profile, save documents with Word into
  a folder and pass it to `scripts/reproduce.sh` (or run `dfp app` yourself);
  half of them become the held-out real-file test set.
* **Govdocs1.** The second test set used a different local corpus (Python,
  Perl and X11 files). To use Govdocs1 as the proposal suggests:
  `python scripts/fetch_govdocs1.py --threads 0 1 --out gd`, then
  `python -m dfp evaluate --corpus <corpus> -o reports/ --second-sources gd`
  (each thread is a 487 MB download).
* **Excel and PowerPoint** are in the producer table as "not yet profiled":
  add Excel- and PowerPoint-saved files with
  `python -m dfp app <corpus> <dir> --name excel` (or `powerpoint`), and record
  their option bits and stream endings in `dfp/producers.py`. Mac builds of
  Word were not measured either.
* **Time-permitting items** (Google Docs exports, APKs built with the Android
  build tools): add them with `dfp app` in the same way.

---

## 8. Limitations

* A profile names the compression library, not the program: every zlib-based
  program (Python, Java, .NET 8, libarchive, LibreOffice) shares one profile,
  and only container metadata separates them.
* Small streams carry little evidence; below the minimum size the tool reports
  "insufficient evidence" instead of guessing, and an exact match counts as
  proof only from 1 KiB and 64 LZ77 matches.
* Rejecting encoders never seen in training is hard when they resemble a known
  one (see the unknown-encoder results): left out of training, zopfli is
  rejected 80% of the time but Chromium's zlib only 8% (mostly attributed to
  zlib-ng) and Word never (attributed to zlib).
* An edit is reported only on exact evidence. An edited part that ends its
  stream the way the original writer does, and is not proven by a reference
  encoder of a different family, is "cannot confirm": for example a zlib-ng
  edit of a zlib-written file, an edit smaller than 1 KiB in a file written by
  a zlib-based program, or any edit when the editor's reference encoder is not
  installed (section 6 gives the rates without reference encoders).
* Word builds differ in their match finder: the Word profile was learnt from
  Word 16 on one Windows PC, and parts of documents from other Word builds may
  be attributed to zlib. That is why a zlib match is never evidence against a
  Word claim; the stream endings and option bits, which every measured Word
  build shares, are what the claim is checked against.
* Genuine files can mix stream endings when two programs really wrote them:
  documents co-edited in Word for the web keep parts written by desktop Word,
  and a ZIP repackaged by another program keeps the original compressed
  parts. The tool reports those as written by more than one program, which is
  true, but not why.
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

```bash
python -m unittest discover -s tests
```

107 tests: `tests/test_dfp.py` (58) covers the bit reader, the parser
against zlib (350 streams across all levels and strategies, plus truncation),
optimal Huffman construction, decision features, every available encoder, the
corpus (unique sources, label sets, Java sharing the zlib profile,
regeneration), calibration and explanations, the vote, producer claims and the
producer table, ZIP rewriting, the baselines, verification by re-encoding,
metadata robustness, signatures, the covert-channel extension, the size-band
reliability rule and the bundled model; `tests/test_review_fixes.py` (49) has a
regression test for every fix in `CHANGELOG.md`: parser strictness, ZIP64,
prepended stubs, encrypted entries, multi-member GZIP, stream endings, proofs
and their limits, mixed-encoder and producer findings, ties, the family split
and the end-to-end metrics. Tests that need an optional encoder or the
preflate-rs tool skip when it is missing.

---

## 11. Repository layout

```
deflate-provenance/
  dfp/
    containers.py   ZIP/ZIP64/OOXML/APK/JAR/ODF/EPUB/GZIP/zlib/PNG/PDF/raw + claims
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
    aggregate.py    archive analysis: proofs, stream endings, consistency findings
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
  scripts/reproduce.sh       regenerates results/ and the bundled model (in the image)
  scripts/fetch_govdocs1.py  Govdocs1 sample downloader
  results/                   the published evaluation (HTML, JSON, manifest)
  tests/test_dfp.py, tests/test_review_fixes.py
  Dockerfile, requirements.txt, CHANGELOG.md
```

---

## 12. Use of generative AI

Per the assignment's disclosure requirement, record in your submission where
AI assistance was used (for example drafting code and documentation). The
correctness and academic integrity of the final work are the team's
responsibility: validate every result independently (the test suite and the
`evaluate` and `demo` outputs are the primary validation here).
