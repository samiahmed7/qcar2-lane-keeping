#!/usr/bin/env bash
# Rebuild large runtime weight files from committed GitHub-safe parts.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="weights/car_track_v3_lane.onnx"
EXPECTED_SHA="c981e0652eb1c268f1fa28f7cc4d51e8df8b9064d78c7a3e3b84dfda0d762fbd"

sha_of() {
    sha256sum "$1" | awk '{print $1}'
}

if [ -f "${MODEL}" ]; then
    current_sha="$(sha_of "${MODEL}")"
    if [ "${current_sha}" = "${EXPECTED_SHA}" ]; then
        echo "OK: ${MODEL} already restored"
        exit 0
    fi
    echo "Existing ${MODEL} has SHA ${current_sha}; rebuilding from parts..."
fi

parts=(weights/car_track_v3_lane.onnx.part-*)
if [ ! -e "${parts[0]}" ]; then
    echo "Missing ${MODEL} and no part files were found."
    exit 1
fi

tmp="${MODEL}.tmp.$$"
rm -f "${tmp}"
cat "${parts[@]}" > "${tmp}"

actual_sha="$(sha_of "${tmp}")"
if [ "${actual_sha}" != "${EXPECTED_SHA}" ]; then
    rm -f "${tmp}"
    echo "Restored ${MODEL} SHA mismatch:"
    echo "  expected ${EXPECTED_SHA}"
    echo "  actual   ${actual_sha}"
    exit 1
fi

mv "${tmp}" "${MODEL}"
echo "Restored ${MODEL}"
