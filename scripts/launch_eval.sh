#!/usr/bin/env bash
set -euo pipefail

TREATMENT="${1:-configs/experiment/main.conf}"
python -m vinexplainnet.studio evaluate --treatment "${TREATMENT}"
python -m vinexplainnet.studio explain --treatment "${TREATMENT}"
