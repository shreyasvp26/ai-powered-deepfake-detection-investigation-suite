#!/usr/bin/env bash
# Happy path against docker compose: POST /v1/jobs, poll until done, show verdict line.
# Usage:  ./scripts/docker-smoke.sh [BASE_URL]   (default: http://127.0.0.1:8000)
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
TMP_DIR="${TMPDIR:-/tmp}"
CLIP="${TMP_DIR}/df_smoke_$$.mp4"

python3 - <<'PY' > "${CLIP}"
import sys
# Minimal ftyp (matches api/tests) + padding; ffprobe returns duration; probe in container uses ffmpeg.
b = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isommp41" + b"\x00" * 4000
sys.stdout.buffer.write(b)
PY

echo "POST ${BASE}/v1/jobs"
RESP=$(curl -fsS -F "file=@${CLIP};type=video/mp4" "${BASE}/v1/jobs")
echo "$RESP" | head -c 300
echo
ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

for i in $(seq 1 45); do
  B=$(curl -fsS "${BASE}/v1/jobs/${ID}")
  ST=$(echo "$B" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  if [ "$ST" = "done" ] || [ "$ST" = "failed" ]; then
    echo "Final status: $ST"
    echo "$B" | python3 -m json.tool | head -n 40
    rm -f "${CLIP}"
    exit 0
  fi
  sleep 1
done
echo "Timeout waiting for job" >&2
rm -f "${CLIP}"
exit 1
