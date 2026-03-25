
# Usage: RM_MODEL_PATH=/path/to/model bash verl_utils/reward/setup.sh
# Environment variables:
#   RM_MODEL_PATH  — path to reward model (default: SB_DAPO_RL_32B checkpoint)

# sudo apt-get install net-tools
# export BYTED_RAY_SERVE_RUN_HOST="::"

export RM_MODEL_PATH="${RM_MODEL_PATH:-/mnt/bn/trae-research-models-lq/xujunjielong/models/R4P-32B}"

pip install -e .

ray stop --force
ray start --head \
  --node-ip-address=127.0.0.1 \
  --dashboard-host=127.0.0.1 \
  --disable-usage-stats
python verl_utils/reward/model_server.py

echo "Server is running. Press Ctrl+C to stop."
while true; do
    sleep 3600
    echo "[Heartbeat] Server is still running at $(date)"
done
# serve shutdown # for exit app