export ROOT_DIR='/mnt/bn/trae-research-models-lq/xujunjielong'
export BASE_MODEL=$ROOT_DIR'/models/Qwen3-32B'
export WAND_PROJECT='verifier-rm'
export EXPERIMENT_NAME='TRAE_R2E'

# ARL environment config
export ARL_GATEWAY_URL="${ARL_GATEWAY_URL:-http://118.145.210.10:8080}"
export ARL_MIRROR_NAMESPACE="${ARL_MIRROR_NAMESPACE:-code}"
export ARL_REWARD_CONCURRENCY="${ARL_REWARD_CONCURRENCY:-16}"
export ARL_REWARD_TIMEOUT="${ARL_REWARD_TIMEOUT:-600}"
export ARL_EXPERIMENT_ID="${ARL_EXPERIMENT_ID:-default}"
# Reward model server URL (for RM-based reward)
export RM_SERVER_URL="${RM_SERVER_URL:-http://[2605:340:cd51:4900:14b1:50d9:ed35:b3f4]:8365/score}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-error}"
export HF_ENDPOINT=https://hf-mirror.com

# ── Clean up stale ARL pods from previous runs ──
echo "Cleaning up ARL pods for experiment '$ARL_EXPERIMENT_ID'..."
cleanup_resp=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
    "${ARL_GATEWAY_URL}/v1/managed/experiments/${ARL_EXPERIMENT_ID}")
echo "ARL cleanup: HTTP $cleanup_resp"

# sleep 5m

git config --global safe.directory "*"
pip install pylint arl-env==0.3.0 swebench

# ── Data preparation: download from HuggingFace and convert to verl format ──
DATA_DIR=$ROOT_DIR/datasets/r2e
mkdir -p $DATA_DIR

# Step 1: Download R2E-Gym Subset + SWE-Bench Verified → info parquets
# (extra_info stored as JSON string, same as rllm's swe_dataset.py)
python3 verl_utils/data/swe_dataset.py --local_dir $DATA_DIR

# Step 2: Convert to verl training format via data_process.py
python3 verl_utils/data/data_process.py --data_source r2e_train --file_path $DATA_DIR/info_r2e_train.parquet
python3 verl_utils/data/data_process.py --data_source r2e_test  --file_path $DATA_DIR/info_r2e_test.parquet

train_files="['$DATA_DIR/data_r2e_train.parquet']"
test_files="['$DATA_DIR/data_r2e_test.parquet']"

mkdir -p /opt/tiger/verifier-rm/workspace
tool_config_path=/opt/tiger/verifier-rm/verl_utils/tool/config/r2egym_tool_config.yaml

# Reward function (override via REWARD_TYPE=arl or REWARD_TYPE=rm)
REWARD_TYPE="${REWARD_TYPE:-rm}"
if [ "$REWARD_TYPE" = "arl" ]; then
    REWARD_PATH=verl_utils/reward/arl_reward.py
    REWARD_NAME=compute_score_arl_clip
else
    REWARD_PATH=verl_utils/reward/model_client.py
    REWARD_NAME=compute_score_remote_clip
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50 \
    +actor_rollout_ref.rollout.multi_turn.max_concurrent_agents=32 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    data.return_raw_chat=True \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.shuffle=True \
    +data.seed=42 \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=28672 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.skip_tokenizer_init=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$ROOT_DIR/rollouts/$EXPERIMENT_NAME \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=$REWARD_NAME \
    reward_model.reward_manager=naive \
    2>&1 | tee $EXPERIMENT_NAME.log
