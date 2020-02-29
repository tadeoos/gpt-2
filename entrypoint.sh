#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

cd /gpt-2
python python3 -X utf8 serve.py
