#!/bin/bash
set -x

# Export CUDA environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# Force SDPA attention (flash-attn not installed)
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

# Use Qwen3-VL-4B model (confirmed to exist and support multimodal)
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct

python3 -m examples.multimodal_geo3k.train_geo3k \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=2 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    rllm.mask_truncated_samples=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='rllm-multimodal' \
    trainer.experiment_name='geo3k-qwen3-vl-4b-sglang' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    rllm.agent.max_steps=1 \
    rllm.stepwise_advantage.enable=False \
    rllm.workflow.use_workflow=False \
    trainer.total_epochs=5
