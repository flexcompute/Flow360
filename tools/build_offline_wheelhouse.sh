#!/usr/bin/env bash
# Build an offline wheelhouse for flow360 targeting a specific (platform, Python).
#
# Produces:
#   <output_dir>/
#     wheelhouse/*.whl          -- every runtime dep wheel + flow360 wheel
#     requirements.txt          -- pinned versions exported from poetry.lock
#
# Usage:
#   tools/build_offline_wheelhouse.sh \
#     --python <python_bin> \
#     --output <bundle_dir> \
#     --python-version <3.10|3.11|3.12|3.13> \
#     --pip-platform <manylinux2014_x86_64|macosx_11_0_arm64|win_amd64|...> \
#     [--pip-platform-fallback <tag>]
#
# --pip-platform explicitly pins the wheel platform tag pip downloads. Using
# pip download --platform ensures Linux wheelhouses are manylinux2014-compatible
# (wide glibc support) even when built on a newer runner.

set -euo pipefail

PYTHON_BIN=""
OUTPUT_DIR=""
PY_VER=""
PIP_PLATFORM=""
PIP_PLATFORM_FALLBACK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)                 PYTHON_BIN="$2"; shift 2 ;;
    --output)                 OUTPUT_DIR="$2"; shift 2 ;;
    --python-version)         PY_VER="$2"; shift 2 ;;
    --pip-platform)           PIP_PLATFORM="$2"; shift 2 ;;
    --pip-platform-fallback)  PIP_PLATFORM_FALLBACK="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$PYTHON_BIN"   ]] && { echo "ERROR: --python required" >&2; exit 2; }
[[ -z "$OUTPUT_DIR"   ]] && { echo "ERROR: --output required" >&2; exit 2; }
[[ -z "$PY_VER"       ]] && { echo "ERROR: --python-version required" >&2; exit 2; }
[[ -z "$PIP_PLATFORM" ]] && { echo "ERROR: --pip-platform required" >&2; exit 2; }

wheelhouse="${OUTPUT_DIR}/wheelhouse"
req_file="${OUTPUT_DIR}/requirements.txt"
mkdir -p "$wheelhouse"

platform_args=(--platform "$PIP_PLATFORM")
if [[ -n "$PIP_PLATFORM_FALLBACK" ]]; then
  platform_args+=(--platform "$PIP_PLATFORM_FALLBACK")
fi

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
cleaned = re.sub(r"^--(extra-)?index-url\s.*\n", "", orig, flags=re.M)
p.write_text(cleaned)
sys.stderr.write(f"stripped index-url directives: {orig.count(chr(10)) - cleaned.count(chr(10))} line(s)\n")
'
"$PYTHON_BIN" -c "$python_strip" "$req_file"

echo "Exported $(wc -l < "$req_file") lines to ${req_file}"
echo "--- first 40 lines ---"
head -40 "$req_file"
echo "::endgroup::"

echo "::group::Download dependency wheels for ${PIP_PLATFORM} py${PY_VER}"
"$PYTHON_BIN" -m pip download \
  --dest "$wheelhouse" \
  --requirement "$req_file" \
  --only-binary=:all: \
  --python-version "$PY_VER" \
  --implementation cp \
  "${platform_args[@]}"
echo "::endgroup::"

echo "::group::Build flow360 wheel (pure-python, platform-agnostic)"
"$PYTHON_BIN" -m pip wheel \
  --wheel-dir "$wheelhouse" \
  --no-deps \
  .
echo "::endgroup::"

echo "::group::Wheelhouse contents"
ls -lh "$wheelhouse"
echo "Total wheels: $(ls "$wheelhouse" | wc -l)"
echo "::endgroup::"
