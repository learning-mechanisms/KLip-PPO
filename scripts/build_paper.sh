#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"
WEBSITE_DIR="$ROOT_DIR/website"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/.build/paper}"
MAIN_TEX="main.tex"
JOB="KLip-PPO"
PAPER_PDF="$PAPER_DIR/$JOB.pdf"
WEBSITE_PDF="$WEBSITE_DIR/$JOB.pdf"

mkdir -p "$BUILD_DIR" "$WEBSITE_DIR"

run_latexmk() {
  (
    cd "$PAPER_DIR"
    latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
      -jobname="$JOB" -outdir="$BUILD_DIR" "$MAIN_TEX"
  )
}

run_tectonic() {
  (
    cd "$PAPER_DIR"
    "$@" --outdir "$BUILD_DIR" "$MAIN_TEX"
  )
}

run_pdflatex_bibtex() {
  (
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname="$JOB" -output-directory "$BUILD_DIR" "$MAIN_TEX"
    (cd "$BUILD_DIR" && BIBINPUTS="$PAPER_DIR:" bibtex "$JOB")
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname="$JOB" -output-directory "$BUILD_DIR" "$MAIN_TEX"
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname="$JOB" -output-directory "$BUILD_DIR" "$MAIN_TEX"
  )
}

if command -v latexmk >/dev/null 2>&1; then
  run_latexmk
elif command -v tectonic >/dev/null 2>&1; then
  run_tectonic tectonic
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  run_pdflatex_bibtex
elif command -v pixi >/dev/null 2>&1; then
  run_tectonic pixi exec -s tectonic tectonic
else
  cat >&2 <<'EOF'
No LaTeX toolchain found. Install latexmk, tectonic, or pdflatex+bibtex.
EOF
  exit 1
fi

if [[ ! -f "$BUILD_DIR/$JOB.pdf" ]]; then
  echo "Expected PDF was not produced: $BUILD_DIR/$JOB.pdf" >&2
  exit 1
fi

cp "$BUILD_DIR/$JOB.pdf" "$PAPER_PDF"
cp "$PAPER_PDF" "$WEBSITE_PDF"

echo "Wrote $PAPER_PDF"
echo "Synced $WEBSITE_PDF"
