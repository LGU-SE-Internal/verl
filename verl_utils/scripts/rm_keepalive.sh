#!/bin/bash
# RM Server Keep-Alive Script
# Sends mock requests (100 batches × 4 patches = 400 items) every 5 minutes
# to maintain GPU utilization and prevent auto-kill.
#
# Usage:
#   RM_SERVER_URL=http://your-rm-server:8365/score bash verl_utils/scripts/rm_keepalive.sh
#   # or with defaults from TRAE_R2E.sh:
#   bash verl_utils/scripts/rm_keepalive.sh

set -euo pipefail

RM_SERVER_URL="${RM_SERVER_URL:-http://[2605:340:cd51:601:ac2e:6a32:7f73:b1a7]:8365/score}"
INTERVAL_SECONDS="${RM_KEEPALIVE_INTERVAL:-300}"  # 5 minutes
NUM_BATCHES="${RM_KEEPALIVE_BATCHES:-500}"         # 500 batches
BATCH_SIZE=4                                       # 4 patches per batch

MOCK_ISSUE="This is a keepalive mock issue to maintain GPU utilization."
MOCK_PATCH="diff --git a/mock.py b/mock.py\n--- a/mock.py\n+++ b/mock.py\n@@ -1 +1 @@\n-pass\n+pass  # keepalive"

# Build the JSON payload once
build_payload() {
    python3 -c "
import json

num_batches = ${NUM_BATCHES}
batch_size = ${BATCH_SIZE}
issue = '''${MOCK_ISSUE}'''
patch = '''${MOCK_PATCH}'''

batches = []
for i in range(num_batches):
    batches.append({
        'batch_id': f'keepalive_{i}',
        'data': {
            'issue': issue,
            'patch_list': [patch] * batch_size
        }
    })

print(json.dumps({'batches': batches}))
"
}

PAYLOAD=$(build_payload)
TOTAL_ITEMS=$((NUM_BATCHES * BATCH_SIZE))

echo "=== RM Keep-Alive Started ==="
echo "  Server:   ${RM_SERVER_URL}"
echo "  Interval: ${INTERVAL_SECONDS}s"
echo "  Payload:  ${NUM_BATCHES} batches × ${BATCH_SIZE} patches = ${TOTAL_ITEMS} items"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    HTTP_CODE=$(curl -s -o /tmp/rm_keepalive_resp.json -w "%{http_code}" \
        -X POST "${RM_SERVER_URL}" \
        -H "Content-Type: application/json" \
        -d "${PAYLOAD}" \
        --max-time 120 \
        -x "" 2>&1) || HTTP_CODE="ERR"

    if [ "${HTTP_CODE}" = "200" ]; then
        echo "[${TIMESTAMP}] Keep-alive OK (HTTP ${HTTP_CODE})"
    else
        RESP=$(cat /tmp/rm_keepalive_resp.json 2>/dev/null || echo "no response body")
        echo "[${TIMESTAMP}] Keep-alive FAILED (HTTP ${HTTP_CODE}): ${RESP}"
    fi

    sleep "${INTERVAL_SECONDS}"
done
