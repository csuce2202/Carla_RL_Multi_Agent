import os
import sys
import argparse
import ray
from pathlib import Path

from configs.env_config import ENV_CONFIG
from configs.agent_config import AGENT_CONFIG
from configs.train_config import TRAIN_CONFIG
from scripts.train import train_agents
from scripts.evaluate import evaluate_agents
from scripts.visualize import visualize_agents


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Multi-Agent RL")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "visualize"],
                        help="Run mode: train, evaluate, or visualize")
    parser.add_argument("--map", type=str, default="Town10HD_Opt",
                        help="CARLA map to use for training/evaluation")
    parser.add_argument("--num-agents", type=int, default=4,
                        help="Number of RL agents to train")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to load model checkpoint (for evaluation or visualization)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="Total timesteps for training")
    return parser.parse_args()


def main():
    args = parse_args()

    # 配置环境
    env_config = ENV_CONFIG.copy()
    env_config["map"] = args.map

    # 配置智能体
    agent_config = AGENT_CONFIG.copy()
    agent_config["num_agents"] = args.num_agents

    # 配置训练参数
    train_config = TRAIN_CONFIG.copy()
    train_config["seed"] = args.seed
    train_config["total_timesteps"] = args.total_timesteps

    # 初始化Ray
    ray.init(ignore_reinit_error=True)

    try:
        if args.mode == "train":
            train_agents(env_config, agent_config, train_config)
        elif args.mode == "evaluate":
            if args.checkpoint_path is None:
                print("Error: checkpoint-path must be provided for evaluation mode")
                sys.exit(1)
            evaluate_agents(env_config, agent_config, train_config, args.checkpoint_path)
        elif args.mode == "visualize":
            if args.checkpoint_path is None:
                print("Error: checkpoint-path must be provided for visualization mode")
                sys.exit(1)
            visualize_agents(env_config, agent_config, train_config, args.checkpoint_path)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()