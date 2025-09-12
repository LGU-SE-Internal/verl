cd /opt/tiger/patch_verifier/verl
pip install -e .
cd /opt/tiger/patch_verifier

sed -i 's|tool_json_path: "/mnt/bn/trae-research-models/xujunjielong/datasets/context"|tool_json_path: "/mnt/bn/trae-research-models-lq/xujunjielong/datasets/context"|g' verl_utils/tool/config/tool_config/agentic_tool_config.yaml

sudo chmod -R 777 /mnt/bn/trae-research-models-lq/xujunjielong

ROOT_DIR='/mnt/bn/trae-research-models-lq/xujunjielong'
export BASE_MODEL=$ROOT_DIR'/models/Qwen2.5-Coder-7B-Instruct'
export WAND_PROJECT='patch_verifier'
export EXPERIMENT_NAME='SB_DAPO_RL_7B_FULL_MORE'

train_files="['$ROOT_DIR/datasets/data_train_batch_full_more.parquet']"
test_files="['$ROOT_DIR/datasets/data_test_batch_without_tool.parquet']"


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.shuffle=True \
    +data.seed=42 \
    data.train_batch_size=128 \
    data.max_prompt_length=24576 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=32 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=8 \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$ROOT_DIR/rollouts/$EXPERIMENT_NAME \
    custom_reward_function.path=verl_utils/reward/reward_fn.py \
    custom_reward_function.name=compute_score \
    2>&1 | tee $EXPERIMENT_NAME.log