# Published evaluation

`evaluation.html` (open in a browser) and `evaluation.json` are the output of
the evaluation described in the proposal (section III.C). `manifest.json` is
the corpus manifest: every source file, every encoder with its exact version
and settings, and the measured profile-sharing rates.

## How they were produced

Reference machine: Ubuntu 24.04, Python 3.11, 4 CPU cores, with the encoder
versions listed in `manifest.json` (and pinned in the `Dockerfile`).

```bash
python -m dfp corpus -o corpus --per-combo 4 \
    --sources /usr/share/doc --sources /usr/share/mime \
    --sources /usr/lib/libreoffice/share/config --sources /usr/share/i18n \
    --sources /usr/share/fonts --source-limit 36

python -m dfp evaluate --corpus corpus -o results \
    --second-sources /usr/lib/python3.11 /usr/share/perl /usr/share/X11 --second-limit 120 \
    --libreoffice 10 --python-docx 20
```

* **Training corpus:** 168 generated sources (7 content types x 6 sizes x 4)
  plus 180 real files from the listed directories, each compressed with every
  encoder and setting.
* **Test split:** 30% of the source files, chosen before training; the report
  confirms that no test stream shares a source file with training.
* **Second test set:** 120 files from a different corpus (Python and Perl
  sources, X11 data), never used for training. Govdocs1 was the proposal's
  example, but downloads were blocked on the reference machine; use
  `scripts/fetch_govdocs1.py` to repeat the run with it.
* **Real application files:** 30 documents generated with LibreOffice
  24.2.7.2 (10 DOCX, 10 ODT, 10 XLSX) and 20 python-docx documents. No
  Microsoft Word files were available; see the main README, section 7.

Generated content, file lists taken from the local system and random forest
seeds are all fixed, so the same command on the same software versions gives
the same numbers.
