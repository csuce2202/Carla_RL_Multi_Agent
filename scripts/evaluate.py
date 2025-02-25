import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from RL_multi.envs.carla_env import CARLAEnv
from RL_multi.agents.agent_utils import evaluate_model


def evaluate_agents(env_config, agent_config, train_config, checkpoint_path):
    """
    评估训练好的RL智能体

    参数:
        env_config: 环境配置
        agent_config: 智能体配置
        train_config: 训练配置
        checkpoint_path: 模型检查点路径
    """
    print(f"开始评估模型: {checkpoint_path}")

    # 创建评估环境
    print("创建评估环境...")
    env_config_eval = env_config.copy()
    env_config_eval['render_mode'] = 'rgb_array'  # 评估时不需要可视化

    env = CARLAEnv(env_config_eval)

    # 设置随机种子
    seed = train_config.get('seed', 42)
    env.reset(seed=seed)

    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    device = "cuda" if train_config.get('device', 'auto') == 'cuda' else "cpu"
    model = PPO.load(checkpoint_path, env=env, device=device)

    # 评估参数
    num_episodes = train_config.get('eval_episodes', 10)
    deterministic = train_config.get('eval_deterministic', True)

    # 开始评估
    print(f"开始评估 {num_episodes} 个回合...")
    eval_results = evaluate_model(model, env, num_episodes, deterministic)

    # 打印评估结果
    print("\n===== 评估结果 =====")
    print(f"平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"平均步数: {eval_results['mean_steps']:.1f}")
    print(f"平均碰撞次数: {eval_results['mean_collisions']:.2f}")
    print(f"平均车道偏离次数: {eval_results['mean_lane_invasions']:.2f}")

    # 保存评估结果
    results_dir = os.path.join(os.path.dirname(checkpoint_path), 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(results_dir, f'eval_results_{timestamp}.json')

    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)

    # 绘制奖励分布图
    plt.figure(figsize=(10, 6))
    plt.hist(eval_results['episode_rewards'], bins=10, alpha=0.7)
    plt.axvline(eval_results['mean_reward'], color='r', linestyle='dashed', linewidth=2,
                label=f'Mean: {eval_results["mean_reward"]:.2f}')
    plt.title('Episode Rewards Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图表
    plot_file = os.path.join(results_dir, f'reward_distribution_{timestamp}.png')
    plt.savefig(plot_file)
    plt.close()

    print(f"评估结果已保存到: {results_file}")
    print(f"奖励分布图已保存到: {plot_file}")

    # 清理环境
    env.close()

    return eval_results


def evaluate_on_maps(env_config, agent_config, train_config, checkpoint_path, maps=None):
    """
    在多个地图上评估模型

    参数:
        env_config: 环境配置
        agent_config: 智能体配置
        train_config: 训练配置
        checkpoint_path: 模型检查点路径
        maps: 要评估的地图列表，如果为None则使用默认地图列表
    """
    if maps is None:
        maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']

    results = {}

    for map_name in maps:
        print(f"\n===== 评估地图: {map_name} =====")

        # 更新环境配置
        env_config_map = env_config.copy()
        env_config_map['map'] = map_name

        # 进行评估
        map_results = evaluate_agents(env_config_map, agent_config, train_config, checkpoint_path)

        # 存储结果
        results[map_name] = {
            'mean_reward': float(map_results['mean_reward']),
            'std_reward': float(map_results['std_reward']),
            'mean_steps': float(map_results['mean_steps']),
            'mean_collisions': float(map_results['mean_collisions']),
            'mean_lane_invasions': float(map_results['mean_lane_invasions'])
        }

    # 保存所有地图的结果
    results_dir = os.path.join(os.path.dirname(checkpoint_path), 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(results_dir, f'multi_map_eval_{timestamp}.json')

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制比较图
    map_names = list(results.keys())
    mean_rewards = [results[m]['mean_reward'] for m in map_names]
    std_rewards = [results[m]['std_reward'] for m in map_names]

    plt.figure(figsize=(12, 6))
    plt.bar(map_names, mean_rewards, yerr=std_rewards, alpha=0.7, capsize=10)
    plt.title('Model Performance Across Different Maps')
    plt.xlabel('Map')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3)

    # 保存图表
    plot_file = os.path.join(results_dir, f'map_comparison_{timestamp}.png')
    plt.savefig(plot_file)
    plt.close()

    print(f"\n所有地图的评估结果已保存到: {results_file}")
    print(f"比较图已保存到: {plot_file}")

    return results


if __name__ == "__main__":
    # 测试评估函数
    from configs.env_config import ENV_CONFIG
    from configs.agent_config import AGENT_CONFIG
    from configs.train_config import TRAIN_CONFIG

    # 假设的模型路径
    checkpoint_path = './checkpoints/carla_rl_test/final_model.zip'

    # 确保路径存在，否则跳过实际评估
    if os.path.exists(checkpoint_path):
        evaluate_agents(ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG, checkpoint_path)

        # 在不同地图上评估
        evaluate_on_maps(ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG, checkpoint_path, ['Town01', 'Town02'])
    else:
        print(f"模型文件不存在: {checkpoint_path}")
        print("跳过评估")