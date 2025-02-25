import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gymnasium import spaces


class CarlaCNN(BaseFeaturesExtractor):
    """
    CNN特征提取器，用于处理CARLA环境的观察数据
    """

    def __init__(self, observation_space, features_dim=256):
        super(CarlaCNN, self).__init__(observation_space, features_dim)

        # 检查观察空间类型
        if isinstance(observation_space, spaces.Dict):
            self.is_dict = True
            # 为每个观察空间创建特征提取网络
            self.extractors = {}

            for key, subspace in observation_space.spaces.items():
                if key == 'camera':
                    # 处理相机图像
                    n_input_channels = subspace.shape[2]  # 通道数，RGB x 堆叠帧数
                    self.extractors[key] = nn.Sequential(
                        nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                    )

                    # 计算CNN输出维度
                    dummy_input = torch.zeros((1,) + subspace.shape)
                    dummy_input = dummy_input.permute(0, 3, 1, 2)  # [B, C, H, W]
                    with torch.no_grad():
                        n_flatten = self.extractors[key](dummy_input).shape[1]

                    self.extractors[key] = nn.Sequential(
                        self.extractors[key],
                        nn.Linear(n_flatten, 128),
                        nn.ReLU()
                    )

                elif key == 'lidar':
                    # 处理激光雷达点云
                    n_points = np.prod(subspace.shape[:-1])  # 点数
                    n_features = subspace.shape[-1]  # 每个点的特征数

                    self.extractors[key] = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(n_points * n_features, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU()
                    )

                else:
                    # 处理低维特征(GNSS, IMU, speedometer等)
                    n_dims = np.prod(subspace.shape)

                    self.extractors[key] = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(n_dims, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU()
                    )

            # 计算组合特征的维度
            self.total_concat_size = 0
            for key in self.extractors:
                if key == 'camera':
                    self.total_concat_size += 128
                elif key == 'lidar':
                    self.total_concat_size += 128
                else:
                    self.total_concat_size += 32

            # 最终的特征提取头
            self.final_layer = nn.Sequential(
                nn.Linear(self.total_concat_size, features_dim),
                nn.ReLU()
            )
        else:
            # 单一观察空间
            self.is_dict = False

            # 处理图像
            if len(observation_space.shape) == 3 and observation_space.shape[2] >= 3:
                n_input_channels = observation_space.shape[2]
                self.cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # 计算CNN输出维度
                dummy_input = torch.zeros((1,) + observation_space.shape)
                dummy_input = dummy_input.permute(0, 3, 1, 2)
                with torch.no_grad():
                    n_flatten = self.cnn(dummy_input).shape[1]

                self.linear = nn.Sequential(
                    nn.Linear(n_flatten, features_dim),
                    nn.ReLU()
                )
            else:
                # 处理其他类型的输入
                n_input = np.prod(observation_space.shape)
                self.linear = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_input, 128),
                    nn.ReLU(),
                    nn.Linear(128, features_dim),
                    nn.ReLU()
                )

    def forward(self, observations):
        """
        前向传播函数
        """
        if self.is_dict:
            # 处理字典观察空间
            encoded_tensor_list = []

            for key, extractor in self.extractors.items():
                observations_key = observations[key]

                if key == 'camera':
                    # 图像需要转置为(B, C, H, W)格式
                    observations_key = observations_key.permute(0, 3, 1, 2)

                encoded_tensor_list.append(extractor(observations_key))

            # 连接所有特征
            concatenated = torch.cat(encoded_tensor_list, dim=1)

            # 最终层
            return self.final_layer(concatenated)
        else:
            # 处理单一观察空间
            if hasattr(self, 'cnn'):
                # 处理图像
                observations = observations.permute(0, 3, 1, 2)
                return self.linear(self.cnn(observations))
            else:
                # 处理其他类型的输入
                return self.linear(observations)


class RLAgent:
    """
    强化学习智能体，用于自动驾驶
    """

    def __init__(self, env, config, seed=None):
        self.env = env
        self.config = config
        self.seed = seed

        # 设置随机种子
        if seed is not None:
            set_random_seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型类型
        self.agent_type = config.get('agent_type', 'ppo')

        # 初始化模型
        self.model = None

    def create_model(self, policy_kwargs=None):
        """创建RL模型"""
        if policy_kwargs is None:
            # 默认策略网络配置
            policy_kwargs = {
                'features_extractor_class': CarlaCNN,
                'features_extractor_kwargs': {'features_dim': 256}
            }

        if self.agent_type == 'ppo':
            self.model = PPO(
                "MultiInputPolicy" if isinstance(self.env.observation_space, spaces.Dict) else "CnnPolicy",
                self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                n_steps=self.config.get('n_steps', 2048),
                batch_size=self.config.get('batch_size', 64),
                n_epochs=self.config.get('n_epochs', 10),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 0.95),
                clip_range=self.config.get('clip_range', 0.2),
                normalize_advantage=self.config.get('normalize_advantage', True),
                ent_coef=self.config.get('ent_coef', 0.01),
                vf_coef=self.config.get('vf_coef', 0.5),
                max_grad_norm=self.config.get('max_grad_norm', 0.5),
                use_sde=self.config.get('use_sde', False),
                sde_sample_freq=self.config.get('sde_sample_freq', -1),
                target_kl=self.config.get('target_kl', None),
                tensorboard_log=self.config.get('tensorboard_log', "./tensorboard_logs/"),
                policy_kwargs=policy_kwargs,
                verbose=self.config.get('verbose', 1),
                seed=self.seed,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")

        return self.model

    def load_model(self, path):
        """加载预训练模型"""
        if self.agent_type == 'ppo':
            self.model = PPO.load(path, env=self.env, device=self.device)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")

        return self.model

    def save_model(self, path):
        """保存模型"""
        if self.model is not None:
            self.model.save(path)

    def train(self, total_timesteps, callback=None):
        """训练模型"""
        if self.model is None:
            self.create_model()

        return self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )

    def predict(self, observation, state=None, deterministic=True):
        """使用模型进行预测"""
        if self.model is None:
            raise ValueError("Model not initialized")

        return self.model.predict(observation, state, deterministic=deterministic)

    def evaluate(self, num_episodes=10, deterministic=True):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("Model not initialized")

        # 重置环境
        obs, _ = self.env.reset()

        # 统计信息
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0

        done = False

        for i in range(num_episodes):
            while not done:
                # 预测动作
                action, _ = self.model.predict(obs, deterministic=deterministic)

                # 执行动作
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 更新统计信息
                current_episode_reward += reward
                current_episode_length += 1

            # 记录本轮结果
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)

            # 重置
            obs, _ = self.env.reset()
            current_episode_reward = 0
            current_episode_length = 0
            done = False

        # 计算评估指标
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_length = float(np.mean(episode_lengths))

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }


def make_env(env_class, env_config, seed=None, rank=0):
    """
    创建环境的辅助函数
    """

    def _init():
        env = env_class(env_config)
        env.reset(seed=seed + rank if seed is not None else None)
        return env

    return _init


def create_vec_env(env_class, env_config, num_envs=1, seed=None):
    """
    创建矢量化环境
    """
    env_fns = [make_env(env_class, env_config, seed, i) for i in range(num_envs)]

    if num_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)