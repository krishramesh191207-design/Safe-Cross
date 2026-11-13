#!/usr/bin/env bash
set -euo pipefail

WHEELS_DIR="wheels"
mkdir -p "$WHEELS_DIR"

while IFS= read -r package; do
    package="${package%%#*}"
    package="$(echo "$package" | xargs)"
    [ -z "$package" ] && continue

    echo "Downloading $package..."
    if ! pip download \
        --dest "$WHEELS_DIR" \
        --platform manylinux2014_aarch64 \
        --python-version 3.11 \
        --only-binary=:all: \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "$package"; then
        echo "ERROR: no matching aarch64/Python 3.11 wheel found for $package" >&2
        exit 1
    fi
done < requirements-pi.txt

echo "Wheels downloaded to $WHEELS_DIR."
