#!/usr/bin/env bash
set -euo pipefail

TREATMENT="${1:-configs/experiment/main.conf}"
python -m vinexplainnet.studio train --treatment "${TREATMENT}"
