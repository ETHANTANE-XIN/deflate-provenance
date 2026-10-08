#!/bin/sh
# Regenerate the published corpus, evaluation and bundled model.
#
# Run inside the reference image (see the Dockerfile):
#
#   docker build -t dfp .
#   docker run --rm -e DFP_WORKERS=16 -v "$PWD/out:/out" -v "$PWD/word:/word:ro" \
#       --entrypoint sh dfp /opt/dfp/scripts/reproduce.sh /out /word
#
# OUT_DIR receives corpus/, results/ (evaluation.html, evaluation.json,
# manifest.json) and default.json.gz (the model shipped as
# dfp/bundled_models/default.json.gz).  WORD_DIR, if given, holds documents
# saved by Microsoft Word: they are added as the 'word' application profile
# (half trains, half is held out), and the held-out half is used as real
# application files in the evaluation.
set -eu

OUT=${1:-/out}
WORD=${2:-}
WORKERS=${DFP_WORKERS:-8}
mkdir -p "$OUT"
cd /opt/dfp

python -m dfp encoders | tee "$OUT/encoders.txt"

# 1. reference corpus: 168 generated sources + up to 36 real files per directory
python -m dfp corpus -o "$OUT/corpus" --per-combo 4 --workers "$WORKERS" \
    --sources /usr/share/doc --sources /usr/share/mime \
    --sources /usr/lib/libreoffice/share/config --sources /usr/share/i18n \
    --sources /usr/share/fonts --source-limit 36

REAL=""
if [ -n "$WORD" ]; then
    # 2. documents saved by Microsoft Word become the 'word' profile
    python -m dfp app "$OUT/corpus" "$WORD" --name word \
        --description "Microsoft Word 16.0 (Windows), documents saved by Word" \
        | tee "$OUT/word_app.txt"
    # the held-out half are real application files for the evaluation
    mkdir -p "$OUT/real"
    python - "$OUT/corpus/manifest.json" "$WORD" "$OUT/real" <<'EOF'
import json, shutil, sys
from pathlib import Path
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for name in manifest["app_profiles"]["word"]["holdout_file_names"]:
    target = Path(sys.argv[3]) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(sys.argv[2]) / name, target)
EOF
    REAL="--real $OUT/real"
fi

# 3. the proposal's evaluation (section III.C)
# shellcheck disable=SC2086
python -m dfp evaluate --corpus "$OUT/corpus" -o "$OUT/results" --workers "$WORKERS" \
    --second-sources /usr/lib/python3.12 /usr/share/perl /usr/share/X11 --second-limit 120 \
    --libreoffice 10 --python-docx 20 $REAL
cp "$OUT/corpus/manifest.json" "$OUT/results/manifest.json"

# 4. the shipped model: every corpus source and the Word training half
python -m dfp train --corpus "$OUT/corpus" -o "$OUT/default.json.gz"
echo "done: $OUT/results and $OUT/default.json.gz"
