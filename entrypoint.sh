#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

cd /gpt-2
python src/app.py
