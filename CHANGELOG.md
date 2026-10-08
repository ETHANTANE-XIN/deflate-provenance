# Changelog

## 2.1.0 (October 2026)

Fixes from an independent code review of 2.0.0. Every fix has a regression
test in `tests/test_review_fixes.py`; the corpus, model and published
evaluation were regenerated with this version in the reference container
(`Dockerfile`, `scripts/reproduce.sh`).

### Forensic conclusions

* **Genuine Microsoft Word files were reported as "Go compress/flate,
  consistent".**  Word ends its parts with an empty stored block (a sync
  flush) and an empty fixed-code final block; in the 2.0.0 corpus only Go
  wrote empty stored blocks, so the forest learnt "empty stored block = Go".
  Block-splitting features now describe only blocks that carry data; flush
  markers and the end-of-stream terminator have features of their own
  (`blk_flush_markers`, `blk_term_empty_stored`, `blk_term_empty_coded`); the
  corpus includes flushed variants of zlib and zlib-ng; and Word is now a
  profile of its own, trained on documents saved by Word 16.
* **"Word never writes zlib" was false.**  Some Word builds (and Excel and
  PowerPoint) write parts that zlib level 1 with a sync flush reproduces byte
  for byte, so 2.0.0's rule would have called genuine Office files
  contradicted.  Word's producer entry now accepts zlib, and what every Word
  build measured does share is checked instead: how its streams end (below).
* **How streams end is the new structural evidence.**  A writer ends every
  DEFLATE stream the same way: a final block that carries data (zlib's
  finish and most encoders), a sync flush and an empty final block
  (Microsoft Office), or an empty stored final block (Go).  This was measured
  on every encoder and ZIP writer in the corpus, 60 Word documents, 30
  LibreOffice documents and about 2,000 ZIP-based files of other origins.
  Entries that end differently were written by different programs, at any
  size and without a model; streams with almost no LZ77 matches (small
  images, audio) are left out because they carry no encoder decisions.  The
  producer table records how each profiled producer ends its parts, so a
  claim of Word on parts that end like zlib's finish (python-docx), or of
  LibreOffice on parts that end like Word's, is contradicted exactly.
* **Mixed-encoder findings rest on exact evidence only:** different stream
  endings, or entries proven by reference encoders of two different families
  with no encoder reproducing both.  Libraries behind the zlib API (zlib,
  zlib-ng, Chromium's zlib) count as one family because they coincide with
  zlib on some inputs: Windows' own `tar.exe` (Chromium's zlib) writes ZIPs
  whose small parts zlib reproduces and whose large parts it does not.
  Statistical disagreement alone, even at high confidence, is "cannot
  confirm".  The entries named are those of the smaller side by entry count;
  a tie names none and says so.  An APK is assembled by copying compressed
  entries from libraries unchanged, and in PDFs, PNGs and GZIP files streams
  from different writers are normal, so there it is reported as information.
* **Exact re-encoding is proof**, and proof needs real encoder decisions.  An
  entry of at least 1 KiB with at least 64 LZ77 matches that CPython zlib
  reproduces byte for byte (81 level/memLevel settings, with and without a
  sync flush) is labelled zlib with confidence 1.0, whatever the forest said
  (2.0.0 computed the match but only reported it).  Below that, encoders
  coincide: every encoder writes incompressible data as the same stored
  blocks, and small Word parts match zlib.  Entries are also re-encoded with
  the leading profile's reference encoder, with those of the model's most
  probable profiles when they match no known profile, and with encoders
  proven elsewhere in the archive.
* **Archive verdict.**  A profile is reported only when entries matching no
  known profile are not the majority by count or by bytes; otherwise the
  verdict is "unknown" and the closest known profile is shown separately.  The
  archive confidence counts bytes that do not support the verdict against it.
* **Claimed-producer contradictions** from the model need an established
  verdict: proven, or every entry above the model's 99%-precision confidence
  with at least 1 KiB of evidence and no exact zlib check against it.
  Producers that write through the zlib API (LibreOffice, openpyxl, the JDK
  `jar` tool) accept zlib, zlib-ng and Chromium's zlib, since the platform
  decides which one they get.  The producer table is applied to ZIP-based
  formats only: a PDF's /Creator names the authoring program, not the PDF
  writer.  A verdict of "consistent" is shown only when nothing could not be
  confirmed; otherwise the report says "inconclusive".
* **Minimum evidence size.**  The held-out calibration fold can measure every
  size band above the reliability target (it did for this corpus, setting the
  minimum to 0 bytes and letting 50-byte parts vote).  The minimum is now
  never below 256 compressed bytes; the measured value is recorded beside it.
* **The report records which reference encoders were missing**, so a result
  no longer silently depends on the machine; application profiles such as
  Word, which have no reference encoder at all, are listed separately rather
  than as "not installed".
* **Duplicate entry names** are reported as inconsistent, and every copy is
  checked against its own bytes (2.0.0 looked entries up by name and
  re-encoded the wrong bytes for later copies).

### Parsing and containers

* Multi-member GZIP files: every member is analysed, each with its own
  CRC32/ISIZE trailer; trailing bytes are reported.
* ZIP64 archives (more than 65,535 entries or offsets beyond 4 GiB) are read
  from the ZIP64 end-of-central-directory record and extra fields.
* ZIPs with unadjusted prepended data (self-extracting stubs) are rebased and
  recognised from their end-of-central-directory record.
* Encrypted entries are listed as "encrypted", never parsed as DEFLATE.
* A stored block whose LEN and NLEN disagree is an error in non-strict mode
  (it used to be accepted silently); incomplete Huffman codes and
  HLIT/HDIST beyond 286/30 are rejected as zlib rejects them.
* Empty blocks no longer allocate per-block histograms (a hostile stream of
  empty blocks cost about 4 KB of memory per input byte).
* Streams larger than 1 MiB compressed are analysed on their first blocks up
  to 1 MiB (training streams are at most 256 KB), and the brute-force zlib
  search stops at the first differing byte: an 8 MB spreadsheet that took
  more than 10 minutes now takes about 30 seconds.
* Entry names with '\\' separators (PowerShell 5.1 Compress-Archive) are
  normalised for claim lookups; JAR manifest continuation lines are joined.
* Decision-feature sampling covers the whole stream (it covered only the
  start of mid-sized streams), and `dis_frac_gt_4k` now counts distances over
  4 KiB (it counted over 2 KiB).
* 7-Zip is found in its Windows install folder and Windows' own `tar.exe`
  (bsdtar) is used as libarchive.

### Classifier

* The "too far from every known profile" rule uses a novelty score on the
  decision features (nearest-neighbour distance to each profile's training
  streams in units of that profile's typical spacing).  The model records how
  often each rule rejects held-out known streams and simulated unseen
  encoders.
* A second, 99%-precision confidence threshold supports mixed-encoder
  findings without exact proof.

### Evaluation

* The train/test split is made by source *family*: files from one directory
  that share a name prefix, or whose content is a near-duplicate, never
  straddle the split.  The overlap checks are independent (content digests
  and near-duplicate sketches), not true by construction.
* A source's id is its content digest, so the same file in two directories
  is one source; the second test set drops files whose content is identical
  to a training source.
* Application profiles added with `dfp app` (Word) are trained on their
  'train' half and tested on their 'holdout' half; `dfp app` lists the
  held-out files (as paths relative to the folder).  Documents saved from one
  template share byte-identical parts (theme, font table), so a held-out
  stream that also occurs in a training document is moved to training, and
  the overlap check counts held-out application streams identical to a
  training stream.
* Both forced-choice and end-to-end metrics are reported.  End to end scores
  what `dfp analyse` reports for each stream, counting "unknown encoder" and
  "insufficient evidence" (below the minimum evidence size) as misses, and is
  the headline; the baseline comparison scores DeflateProvenance the same
  way.
* "Edited entry singled out" now requires the report to name the edited
  entry and no other (2.0.0 never checked which entries were named).
* The metadata baseline is scored at profile level (the same task as
  DeflateProvenance) as well as writer level; brute-force zlib is also scored
  as a zlib detector (precision and recall).
* Leave-one-encoder-out uses the same number of trees as the shipped model and
  reports which rule rejected each unseen stream.
* Archives and real files are analysed exactly as `dfp analyse` does (exact
  re-encoding on); the real-file edit test picks an installed non-zlib
  editor instead of requiring Go; the producer-rewrite test rewrites each
  claim to one it contradicts (Word documents to LibreOffice, others to
  Word) with that producer's option bits.
* The real-file edit test skips a file when the editor's encoder reproduces
  the original part exactly (decided from the bytes, not from the model):
  recompressing a zlib-written LibreOffice part with zlib reproduces exactly
  what LibreOffice writes, which no method can detect, and counting those as
  misses understated detection (they are reported as skipped).  The edit
  always targets the file's largest part.
* The archive experiments also report false alarms and edit detection with
  reference encoders switched off (only CPython zlib's exact checks, as on a
  machine with nothing else installed), skip the impersonation test instead
  of crashing when only one ZIP writer is installed, and count edits skipped
  for want of an editor of another profile.  `--cross-editor` is checked
  before the evaluation starts.
* The synthetic test archives hold five source files as DOCX parts (plus
  `[Content_Types].xml` and `docProps/app.xml`) instead of three: with the
  256-byte minimum, three-part archives often had only two analysable
  entries, and "which of two entries was edited" has no answer.
* `evaluation.json` lists every genuine archive that was flagged and every
  edit that was missed or not singled out, with the findings that fired, and
  the HTML report shows them.

### Tooling

* `dfp train` refuses to replace the 'default' model with one that knows
  fewer profiles than the bundled model (use `--save-as NAME` or `--force`).
* The producer findings print the real `dfp app` command.
* The demo labels its LibreOffice stand-in as written by the demo when
  LibreOffice is not installed, and the mixed-encoder case uses any installed
  non-zlib writer or editor.
* The `Dockerfile` builds: it installs CA certificates before switching to
  the HTTPS-only Ubuntu snapshot, pins the base image digest and every
  package version, verifies the Go and Node.js downloads by SHA-256, pins the
  Rust toolchain, keeps package documentation (corpus sources), and ships
  `scripts/reproduce.sh`, the single command that regenerates `results/` and
  the bundled model.
