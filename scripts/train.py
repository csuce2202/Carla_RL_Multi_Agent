import os
import time
import ray
from pathlib import Path
import numpy as np
import torch

from RL_multi.envs.carla_env import CARLAEnv
from RL_multi.agents.rl_agent import RLAgent, create_vec_env
from RL_multi.agents.agent_utils import create_callback, wrap_env_monitor
from RL_multi.models.ppo_model import create_ppo_model


def train_agents(env_config, agent_config, train_config):
    """
    训练多智能体RL模型

    参数:
        env_config: 环境配置
        agent_config: 智能体配置
        train_config: 训练配置
    """
    print("开始训练多智能体RL模型...")

    # 设置随机种子
    seed = train_config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 设置日志路径
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(train_config.get('tensorboard_log', './tensorboard_logs/'), f'carla_rl_{current_time}')
    os.makedirs(log_dir, exist_ok=True)

    # 设置检查点路径
    checkpoint_path = os.path.join(train_config.get('checkpoint_path', './checkpoints/'), f'carla_rl_{current_time}')
    os.makedirs(checkpoint_path, exist_ok=True)

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建环境
    print("创建CARLA环境...")

    # 设置环境数量
    n_envs = train_config.get('n_envs', 1)

    if n_envs > 1:
        # 使用多进程并行环境
        print(f"创建 {n_envs} 个并行环境...")

        # 为并行环境调整端口
        parallel_envs = []
        for i in range(n_envs):
            env_cfg = env_config.copy()
            env_cfg['carla_port'] = env_config.get('carla_port', 2000) + i
            env_cfg['num_agents'] = 1  # 每个环境只有一个智能体
            parallel_envs.append(env_cfg)

        # 创建并行环境
        vec_env = create_vec_env(CARLAEnv, parallel_envs[0], n_envs, seed)
    else:
        # 单个环境
        print("创建单个环境...")
        env = CARLAEnv(env_config)
        env = wrap_env_monitor(env, log_dir)
        vec_env = create_vec_env(CARLAEnv, env_config, 1, seed)

    # 创建回调
    callbacks = create_callback(
        train_config.get('checkpoint_freq', 10000),
        checkpoint_path,
        log_dir,
        train_config.get('verbose', 1)
    )

    # 创建模型
    print("创建PPO模型...")
    model = create_ppo_model(vec_env, train_config, seed, device)

    # 开始训练
    print(f"开始训练，总步数: {train_config.get('total_timesteps', 1000000)}...")
    model.learn(
        total_timesteps=train_config.get('total_timesteps', 1000000),
        callback=callbacks
    )

    # 保存最终模型
    final_model_path = os.path.join(checkpoint_path, 'final_model')
    print(f"保存最终模型到 {final_model_path}...")
    model.save(final_model_path)

    # 清理环境
    vec_env.close()

    print("训练完成!")
    return final_model_path


if __name__ == "__main__":
    # 测试训练函数
    from configs.env_config import ENV_CONFIG
    from configs.agent_config import AGENT_CONFIG
    from configs.train_config import TRAIN_CONFIG

    # 修改配置以加快测试
    test_env_config = ENV_CONFIG.copy()
    test_agent_config = AGENT_CONFIG.copy()
    test_train_config = TRAIN_CONFIG.copy()
    test_train_config['total_timesteps'] = 1000  # 仅用于测试

    train_agents(test_env_config, test_agent_config, test_train_config)