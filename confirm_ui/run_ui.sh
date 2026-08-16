#!/usr/bin/env bash
# Script to launch the QA Answer Comparison Dashboard
# Operates purely inside confirm_ui

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " Starting QA Answer Comparison UI Server"
echo " Access dashboard at: http://localhost:8000"
echo "=================================================="

python3 server.py "$@"
