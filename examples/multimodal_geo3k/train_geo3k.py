"""
Train a multimodal VL agent on the Geometry3K dataset.

This example demonstrates multimodal support in rLLM by training on
geometry problems with images using models like Qwen2.5-VL or Qwen3-VL.
"""

import hydra

from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_fn import math_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # Load the geometry3k dataset (must run prepare_geo3k_dataset.py first)
    train_dataset = DatasetRegistry.load_dataset("geometry3k", "train")
    test_dataset = DatasetRegistry.load_dataset("geometry3k", "test")

    env_args = {"reward_fn": math_reward_fn}

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={},
        env_args=env_args,
        env_class=SingleTurnEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
