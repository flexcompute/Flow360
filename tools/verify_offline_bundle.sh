#!/usr/bin/env bash
# Verify that a wheelhouse can install flow360 with no index access.
#
# Creates a fresh venv, installs flow360 using --no-index --find-links=<wheelhouse>,
# then imports flow360 as a smoke test. Mirrors exactly what an end user would run.
#
# Usage:
#   tools/verify_offline_bundle.sh --python <python_bin> --wheelhouse <dir>

set -euo pipefail

PYTHON_BIN=""
WHEELHOUSE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)      PYTHON_BIN="$2"; shift 2 ;;
    --wheelhouse)  WHEELHOUSE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$PYTHON_BIN" ]] && { echo "ERROR: --python required" >&2; exit 2; }
[[ -z "$WHEELHOUSE" ]] && { echo "ERROR: --wheelhouse required" >&2; exit 2; }
[[ -d "$WHEELHOUSE" ]] || { echo "ERROR: wheelhouse dir not found: $WHEELHOUSE" >&2; exit 2; }

venv_root="$(mktemp -d)"
venv_dir="${venv_root}/venv"

echo "::group::Create isolated venv at ${venv_dir}"
"$PYTHON_BIN" -m venv "$venv_dir"
echo "::endgroup::"

# Cross-platform venv interpreter path (POSIX vs Windows)
if [[ -x "${venv_dir}/bin/python" ]]; then
  VENV_PY="${venv_dir}/bin/python"
elif [[ -x "${venv_dir}/Scripts/python.exe" ]]; then
  VENV_PY="${venv_dir}/Scripts/python.exe"
else
  echo "ERROR: cannot find venv python under ${venv_dir}" >&2
  exit 1
fi

echo "::group::Install flow360 fully offline from ${WHEELHOUSE}"
"$VENV_PY" -m pip install \
  --no-index \
  --find-links "$WHEELHOUSE" \
  flow360
echo "::endgroup::"

echo "::group::Smoke import"
"$VENV_PY" -c "import flow360; print('flow360 imported OK from', flow360.__file__)"
echo "::endgroup::"

echo "Offline bundle verification PASSED."
