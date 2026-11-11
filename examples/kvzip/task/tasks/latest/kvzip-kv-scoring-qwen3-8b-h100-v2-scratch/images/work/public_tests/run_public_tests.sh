#!/usr/bin/env bash
# Public tests on this container's GPU.  Usage:
#   /workspace/public_tests/run_public_tests.sh [--tier one|quick|full] [--submission DIR] [--version N]
#       [--scorer-dir DIR] [--workload NAME]... [--show-answers N]
set -euo pipefail
exec python3 "$(dirname "$0")/public_runner.py" "$@"
