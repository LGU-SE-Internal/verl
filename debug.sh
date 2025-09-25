cd /opt/tiger/'verifier-rm'
pip install -e .

export VLLM_USE_V1=1
export GLOO_SOCKET_IFNAME='eth0'
export NCCL_SOCKET_IFNAME='eth0'

export ROOT_DIR='/mnt/bn/trae-research-models/xujunjielong'
export BASE_MODEL=$ROOT_DIR'/models/Qwen3-8B'
export WAND_PROJECT='verifier-rm'
export EXPERIMENT_NAME='TRAE_Lite_LOOP'

git config --global safe.directory "*"
pip install pylint

train_files="['$ROOT_DIR/datasets/data_train_gen_async.parquet']"
test_files="['$ROOT_DIR/datasets/data_test_gen_async.parquet']"

mkdir /opt/tiger/'verifier-rm'/workspace
tool_config_path=/opt/tiger/'verifier-rm'/verl_utils/tool/config/lite_tool_config.yaml

# sed -i 's#tool_format_flags = \[extract_tool_format(sol) for sol in solution_strs\]#tool_format_flags = [True for sol in solution_strs]#' verl_utils/reward/model_client.py

# DEFAULT TO FILTER COMPACTS
# +actor_rollout_ref.rollout.enable_compact_filtering=true \

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=loop \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-length-norm" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=32000 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    data.return_raw_chat=True \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.shuffle=True \
    +data.seed=42 \
    data.train_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=28672 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.skip_tokenizer_init=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.default_local_dir=$ROOT_DIR/experiments/verl/$EXPERIMENT_NAME \
    trainer.rollout_data_dir=$ROOT_DIR/rollouts/$EXPERIMENT_NAME \
    custom_reward_function.path=verl_utils/reward/model_client.py \
    custom_reward_function.name=compute_score_remote_clip \
    reward_model.reward_manager=batch \
    2>&1 | tee $EXPERIMENT_NAME.log