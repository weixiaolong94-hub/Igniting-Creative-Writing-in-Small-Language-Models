#!/bin/bash
set -x

export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files= \
    data.val_files= \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.return_raw_chat=True \
    data.truncation='error' \
    actor_rollout_ref.model.path= \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=custom \
    reward_model.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty='low_var_kl' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='' \
    trainer.experiment_name='' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=3000 \
    trainer.test_freq=-1 \
    trainer.total_epochs=5 $@