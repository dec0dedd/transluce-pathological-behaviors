#!/bin/bash
export PYTHONPATH=".:$PYTHONPATH"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_RUNTIME_ENV_IGNORE_CONTENT_HASH_FOR_LOCAL_DIR=1
export RAY_memory_monitor_refresh_ms=0

# Keep the working ptxas fix
cp /root/transluce-pathological-behaviors/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas /usr/local/bin/ptxas
chmod +x /usr/local/bin/ptxas
export TRITON_PTXAS_PATH="/usr/local/bin/ptxas"

#export WANDB_API_KEY="KEY"

# Standard Ray / W&B Setup
export PATH="/usr/local/bin:$PATH"
export RAY_TMPDIR="/root/r"
export RAY_worker_grace_period_ms=10000

# Directory Prep
mkdir -p /root/r
chmod 777 /root/r
mkdir -p /root/transluce-pathological-behaviors/.venv/lib/python3.12/site-packages/pyairports
touch /root/transluce-pathological-behaviors/.venv/lib/python3.12/site-packages/pyairports/__init__.py
echo "AIRPORT_LIST = []" > /root/transluce-pathological-behaviors/.venv/lib/python3.12/site-packages/pyairports/airports.py

# This runs invisibly in the background. 
# It checks the Ray folder every 0.5 seconds and forces wandb-core to be executable.
(while true; do find /root/r/ray/ -name "wandb-core" -exec chmod +x {} + 2>/dev/null; sleep 0.5; done) &
WATCHER_PID=$!

# - 
uv run python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=prepared/train.parquet \
    data.val_files=prepared/train.parquet \
    data.train_batch_size=1 \
    trainer.balance_batch=True \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="HuggingFaceTB/SmolLM2-135M-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    custom_reward_function.path="src/reward_wrapper.py" \
    custom_reward_function.name="compute_score" \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='transluce_test' \
    trainer.experiment_name='smollm_test_run' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    critic.model.path=null \
    critic.model.tokenizer_path=null

kill $WATCHER_PID
uv run wandb sync