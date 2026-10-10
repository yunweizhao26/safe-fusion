#!/bin/bash
set -euo pipefail
.venv/bin/python scripts/analyses/transductive_references/verify_results.py
