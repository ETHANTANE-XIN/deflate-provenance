# Published evaluation

`evaluation.html` (open in a browser) and `evaluation.json` are the output of
the evaluation described in the proposal (section III.C), produced by
DeflateProvenance 2.1.0. `manifest.json` is the corpus manifest: every source
file, every encoder with its exact version and settings, the measured
profile-sharing rates, and the Word application profile (which documents were
held out).

## How they were produced

In the reference image built from this repository's `Dockerfile`: Ubuntu 24.04
pinned by digest, packages frozen to the Ubuntu snapshot of 27 September 2026,
Python 3.12.3 with numpy 2.5.3, Go 1.24.7, Node.js 22.22.2, OpenJDK 21.0.12,
.NET SDK 8.0.131, 7-Zip 23.01, libarchive 3.7.2, LibreOffice 24.2.7 and the
Rust toolchain 1.98.0 (preflate-rs 0.7.6). Every encoder's exact version is in
`manifest.json` and in the encoder table of `evaluation.html`. The run used 16
worker processes; worker count does not change the results.

One command regenerates everything (corpus, Word profile, evaluation and the
bundled model):

```bash
docker build -t dfp .
```

```bash
docker run --rm -e DFP_WORKERS=16 -v "$PWD/out:/out" -v "$PWD/word:/word:ro" --entrypoint sh dfp /opt/dfp/scripts/reproduce.sh /out /word
```

`word/` holds documents saved by Microsoft Word (the published run used 60
documents saved by Word 16 on Windows; they are not in the repository because
Word records the author's account name in them, see the main README,
section 7). Without a Word folder the script still runs, without the Word
profile and the real Word files. The script runs, in order:

1. `dfp corpus` on 168 generated sources (7 content types x 6 sizes x 4) plus
   up to 36 real files from each of `/usr/share/doc`, `/usr/share/mime`,
   `/usr/lib/libreoffice/share/config`, `/usr/share/i18n` and
   `/usr/share/fonts`, each compressed with every encoder and setting;
2. `dfp app ... --name word`: the Word documents, half for training and half
   held out as real application files (the held-out names are in
   `manifest.json`);
3. `dfp evaluate`, with the second test set taken from `/usr/lib/python3.12`,
   `/usr/share/perl` and `/usr/share/X11` (120 files, never used for
   training), 10 LibreOffice documents per format and 20 python-docx
   documents generated in the image, and the held-out Word documents;
4. `dfp train` on the whole corpus, which is the bundled model
   (`dfp/bundled_models/default.json.gz`).

* **Test split:** 30% of the source *families* (files from one directory that
  share a name prefix, or whose content is a near-duplicate, form one
  family), chosen before training. The report checks independently that no
  test source is identical or near-identical to a training source, and that
  no held-out Word stream is byte-identical to a training stream.
* **Second test set:** a different corpus (Python and Perl sources, X11 data).
  Govdocs1 was the proposal's example; use `scripts/fetch_govdocs1.py` and
  `--second-sources` to repeat the run with it.
* **Real application files:** the 30 held-out Word documents, 30 documents
  generated with LibreOffice 24.2.7.2 (10 DOCX, 10 ODT, 10 XLSX) and 20
  python-docx documents.

Generated content, file lists taken from the image and random-forest seeds are
all fixed, so the same command on the same image gives the same numbers.
