#!/usr/bin/env bash

set -euo pipefail

for test_file in tests/*.mojo; do
    printf '\n==> %s\n' "$test_file"
    mojo run -I . "$test_file"
done

printf '\nAll Mojo tests passed.\n'
