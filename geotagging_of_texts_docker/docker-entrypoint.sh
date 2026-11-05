#!/bin/sh
set -eu
mode="CPU"
python -c "import torch,sys; sys.exit(not torch.cuda.is_available())" 2>/dev/null && mode="GPU"
printf '\n==============================================\n'
printf ' Geotagging of Texts (%s)\n' "$mode"
printf ' Loading models, please wait...\n'
printf '==============================================\n\n'
exec uvicorn app.main:app --host 0.0.0.0 --port "$APP_PORT" --no-server-header
