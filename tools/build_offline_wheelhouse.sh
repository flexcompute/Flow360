#!/usr/bin/env bash
# Build an offline wheelhouse for flow360 using the current runner's Python.
#
# Produces:
#   <output_dir>/
#     wheelhouse/*.whl          -- every runtime dep wheel + flow360 wheel
#     requirements.txt          -- pinned versions exported from poetry.lock
#
# Usage:
#   tools/build_offline_wheelhouse.sh --python <python_bin> --output <bundle_dir>
#
# Uses `pip wheel` which transparently handles both wheel-only deps and
# sdist-only deps (building the latter locally into a wheel). The resulting
# wheelhouse contains wheels tagged for the runner's native platform:
#
#   Linux x86_64 / aarch64  -> manylinux_2_28 (glibc >= 2.28)
#   macOS x86_64 / arm64    -> macosx_*
#   Windows x86_64          -> win_amd64
#
# Pure-python deps (including any locally-built sdists) land as py3-none-any.

set -euo pipefail

PYTHON_BIN=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$PYTHON_BIN" ]] && { echo "ERROR: --python required" >&2; exit 2; }
[[ -z "$OUTPUT_DIR" ]] && { echo "ERROR: --output required" >&2; exit 2; }

wheelhouse="${OUTPUT_DIR}/wheelhouse"
req_file="${OUTPUT_DIR}/requirements.txt"
mkdir -p "$wheelhouse"

echo "::group::Python and pip versions"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip --version
echo "::endgroup::"

echo "::group::Bootstrap pip / wheel / poetry"
"$PYTHON_BIN" -m pip install --upgrade pip wheel
"$PYTHON_BIN" -m pip install 'poetry>=1.8' poetry-plugin-export
echo "::endgroup::"

echo "::group::Export pinned runtime requirements from poetry.lock"
"$PYTHON_BIN" -m poetry export \
  --without-hashes \
  --only main \
  --format requirements.txt \
  --output "$req_file"

# Strip any index-url directives poetry may have emitted for private sources
# (e.g. CodeArtifact). Credentials live in the PIP_EXTRA_INDEX_URL env var
# instead, so requirements.txt stays clean and safe to ship inside the bundle.
python_strip='
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
orig = p.read_text()
cleaned = re.sub(r"^--(extra-)?index-url[ \t].*\n", "", orig, flags=re.M)
p.write_text(cleaned)
sys.stderr.write(f"stripped index-url directives: {orig.count(chr(10)) - cleaned.count(chr(10))} line(s)\n")
'
"$PYTHON_BIN" -c "$python_strip" "$req_file"

echo "Exported $(wc -l < "$req_file") lines to ${req_file}"
echo "--- first 40 lines ---"
head -40 "$req_file"
echo "::endgroup::"

echo "::group::Build dependency wheelhouse"
"$PYTHON_BIN" -m pip wheel \
  --wheel-dir "$wheelhouse" \
  --requirement "$req_file"
echo "::endgroup::"

echo "::group::Build flow360 wheel"
"$PYTHON_BIN" -m pip wheel \
  --wheel-dir "$wheelhouse" \
  --no-deps \
  .
echo "::endgroup::"

echo "::group::Wheelhouse contents"
ls -lh "$wheelhouse"
echo "Total wheels: $(ls "$wheelhouse" | wc -l)"
echo "::endgroup::"
