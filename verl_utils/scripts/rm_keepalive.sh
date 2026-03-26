#!/bin/bash
# RM Server Keep-Alive Script
# Usage:
#   RM_SERVER_URL=http://your-rm-server:8365/score bash verl_utils/scripts/rm_keepalive.sh
exec python3 "$(dirname "$0")/rm_keepalive.py" "$@"
