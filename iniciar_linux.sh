#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -e .
test -f .env || cp .env.example .env
printf '%s\n' 'Abra http://localhost:8000. Mantenha o processo aberto; fechar a aba nao interrompe a simulacao.'
exec .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000 --env-file .env
