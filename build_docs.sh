#!/usr/bin/env bash

set -euo pipefail

docs_source="./docs/src"
docs_build="./docs/build"

sphinx-build -W -b doctest "$docs_source" "$docs_build/doctest"

if [[ "${1:-}" == "--build-only" ]]; then
    sphinx-build -W -b html "$docs_source" "$docs_build/html"
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "usage: $0 [--build-only]" >&2
    exit 2
fi

sphinx-autobuild -M html "$docs_source" "$docs_build"
