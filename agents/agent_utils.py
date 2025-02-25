import os
import numpy as np
import torch
import gym
import time
from typing import Dict, List, Any
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    保存在训练期间达到最佳奖励的模型的回调
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -float('inf')

    def _init_callback(self):
        # 创建保存文件夹
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # 获取监控器信息
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # 计算平均奖励
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                # 如果奖励有改善，保存模型
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(os.path.join(self.save_path, 'best_model'))

        return True


class TensorboardCallback(BaseCallback):
    """
    自定义Tensorboard回调，用于记录额外的日志信息
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        # 记录附加信息
        # 这里可以获取环境的额外信息并记录
        # 例如：车速、碰撞次数、车道偏离次数等
        try:
            if hasattr(self.training_env, 'get_attr'):
                # 矢量化环境
                info = self.training_env.get_attr('last_info')[0]
                episode_info = self.training_env.get_attr('episode_rewards')[0]

                if isinstance(info, dict):
                    for key, value in info.items():
                        if isinstance(value, (int, float)):
                            self.logger.record(f"env_info/{key}", value)

                if len(episode_info) > 0:
                    self.logger.record("rollout/ep_rew_mean", np.mean(episode_info[-100:]))
            else:
                # 非矢量化环境
                info = self.training_env.last_info

                if isinstance(info, dict):
                    for key, value in info.items():
                        if isinstance(value, (int, float)):
                            self.logger.record(f"env_info/{key}", value)
        except:
            pass

        return True


def create_callback(checkpoint_freq, checkpoint_path, log_dir=None, verbose=1):
    """
    创建训练回调
    """
    callbacks = []

    # 检查点回调
    if checkpoint_freq > 0 and checkpoint_path is not None:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_path,
            name_prefix='rl_model',
            verbose=verbose
        )
        callbacks.append(checkpoint_callback)

    # 最佳模型保存回调
    if log_dir is not None:
        save_best_callback = SaveOnBestTrainingRewardCallback(
            check_freq=1000,
            log_dir=log_dir,
            verbose=verbose
        )
        callbacks.append(save_best_callback)

    # Tensorboard回调
    callbacks.append(TensorboardCallback())

    return callbacks


def preprocess_lidar(lidar_data):
    """
    预处理LiDAR数据以确保正确的形状

    参数:
        lidar_data: 原始LiDAR数据

    返回:
        处理后的LiDAR数据，形状为(32, 1024, 3)
    """
    import numpy as np

    # 检查输入形状
    if len(lidar_data.shape) == 2 and lidar_data.shape[1] == 3:
        # 当接收到形状为(N, 3)的数据时，需要重塑为(32, 1024, 3)

        # 确定点的数量
        n_points = lidar_data.shape[0]

        # 如果点数太少，进行填充
        if n_points < 32 * 1024:
            # 创建填充数组
            padding = np.zeros((32 * 1024 - n_points, 3))
            lidar_data = np.vstack([lidar_data, padding])
        # 如果点数太多，进行裁剪
        elif n_points > 32 * 1024:
            lidar_data = lidar_data[:32 * 1024]

        # 重塑为(32, 1024, 3)
        lidar_data = lidar_data.reshape(32, 1024, 3)

    # 如果已经是正确的形状，不做处理
    elif lidar_data.shape == (32, 1024, 3):
        pass
    else:
        print(f"警告: 意外的LiDAR数据形状: {lidar_data.shape}, 尝试调整")
        # 尝试调整为期望的形状
        total_points = np.prod(lidar_data.shape) // 3
        if total_points >= 32 * 1024:
            # 展平并重塑
            flattened = lidar_data.reshape(-1, 3)
            lidar_data = flattened[:32 * 1024].reshape(32, 1024, 3)
        else:
            # 创建空数组并填充可用数据
            reshaped = np.zeros((32, 1024, 3))
            flattened = lidar_data.reshape(-1, 3)
            points_to_copy = min(flattened.shape[0], 32 * 1024)
            reshaped.reshape(-1, 3)[:points_to_copy] = flattened[:points_to_copy]
            lidar_data = reshaped

    return lidar_data

def plot_learning_curve(log_folder, title='Learning Curve'):
    """
    绘制学习曲线
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')

    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.savefig(os.path.join(log_folder, 'learning_curve.png'))
    plt.close()


def preprocess_observation(observation, observation_space):
    """
    预处理观察数据
    """
    if isinstance(observation_space, gym.spaces.Dict):
        # 字典观察空间
        processed_obs = {}

        for key, space in observation_space.spaces.items():
            if key in observation:
                if isinstance(space, gym.spaces.Box):
                    # 对Box空间进行归一化
                    if space.low.min() >= 0 and space.high.max() <= 255:
                        # 图像数据，归一化到[0, 1]
                        processed_obs[key] = observation[key].astype(np.float32) / 255.0
                    else:
                        # 其他数值数据，线性缩放到[-1, 1]
                        processed_obs[key] = 2.0 * (observation[key] - space.low) / (space.high - space.low) - 1.0
                else:
                    # 其他空间类型
                    processed_obs[key] = observation[key]

        return processed_obs
    else:
        # 单一观察空间
        if isinstance(observation_space, gym.spaces.Box):
            # 对Box空间进行归一化
            if observation_space.low.min() >= 0 and observation_space.high.max() <= 255:
                # 图像数据，归一化到[0, 1]
                return observation.astype(np.float32) / 255.0
            else:
                # 其他数值数据，线性缩放到[-1, 1]
                return 2.0 * (observation - observation_space.low) / (
                            observation_space.high - observation_space.low) - 1.0
        else:
            # 其他空间类型
            return observation


def wrap_env_monitor(env, log_dir):
    """
    包装环境以使用Monitor记录数据
    """
    os.makedirs(log_dir, exist_ok=True)
    return Monitor(env, log_dir)


def evaluate_model(model, env, num_episodes=10, deterministic=True, device=None):
    """
    评估模型在环境中的表现

    参数:
        model: 强化学习模型
        env: 环境
        num_episodes: 评估轮数
        deterministic: 是否使用确定性动作
        device: 模型所在设备

    返回:
        dict: 评估结果
    """
    import numpy as np
    print(f"模型运行在设备: {device}")

    # 初始化结果统计
    episode_rewards = []
    episode_steps = []
    episode_collisions = []
    episode_lane_invasions = []

    # 观测预处理函数
    def preprocess_observation(obs):
        """根据需要处理观测"""

        # 如果观测是字典类型
        if isinstance(obs, dict):
            processed_obs = obs.copy()

            # 处理LiDAR数据
            if 'lidar' in processed_obs:
                processed_obs['lidar'] = preprocess_lidar(processed_obs['lidar'])

            return processed_obs

        return obs

    def preprocess_lidar(lidar_data):
        """处理LiDAR数据确保形状正确"""

        # 检查输入形状
        if len(lidar_data.shape) == 2 and lidar_data.shape[1] == 3:
            # 当接收到形状为(N, 3)的数据时，需要重塑为(32, 1024, 3)

            # 确定点的数量
            n_points = lidar_data.shape[0]

            # 如果点数太少，进行填充
            if n_points < 32 * 1024:
                # 创建填充数组
                padding = np.zeros((32 * 1024 - n_points, 3))
                lidar_data = np.vstack([lidar_data, padding])
            # 如果点数太多，进行裁剪
            elif n_points > 32 * 1024:
                lidar_data = lidar_data[:32 * 1024]

            # 重塑为(32, 1024, 3)
            lidar_data = lidar_data.reshape(32, 1024, 3)

        # 如果已经是正确的形状，不做处理
        elif lidar_data.shape == (32, 1024, 3):
            pass
        else:
            print(f"警告: 意外的LiDAR数据形状: {lidar_data.shape}, 尝试调整")
            # 尝试调整为期望的形状
            total_points = np.prod(lidar_data.shape) // 3
            if total_points >= 32 * 1024:
                # 展平并重塑
                flattened = lidar_data.reshape(-1, 3)
                lidar_data = flattened[:32 * 1024].reshape(32, 1024, 3)
            else:
                # 创建空数组并填充可用数据
                reshaped = np.zeros((32, 1024, 3))
                flattened = lidar_data.reshape(-1, 3)
                points_to_copy = min(flattened.shape[0], 32 * 1024)
                reshaped.reshape(-1, 3)[:points_to_copy] = flattened[:points_to_copy]
                lidar_data = reshaped

        return lidar_data

    for i in range(num_episodes):
        # 重置环境
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        collisions = 0
        lane_invasions = 0

        # 预处理初始观测
        obs = preprocess_observation(obs)

        while not (done or truncated):
            # 获取模型预测的动作
            action, _ = model.predict(obs, deterministic=deterministic)

            # 如果动作是在CUDA上，将其移到CPU再转换为NumPy
            if hasattr(action, 'device') and str(action.device) != 'cpu':
                action = action.cpu().numpy()

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)

            # 预处理观测
            obs = preprocess_observation(obs)

            # 更新统计信息
            total_reward += reward
            steps += 1

            # 记录碰撞和车道偏离
            if 'collision' in info and info['collision']:
                collisions += 1
            if 'lane_invasion' in info and info['lane_invasion']:
                lane_invasions += 1

        # 记录本回合统计
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_collisions.append(collisions)
        episode_lane_invasions.append(lane_invasions)

        print(f"Episode {i + 1}/{num_episodes}, Reward: {total_reward:.2f}, Steps: {steps}")

    # 计算平均统计
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_steps = np.mean(episode_steps)
    mean_collisions = np.mean(episode_collisions)
    mean_lane_invasions = np.mean(episode_lane_invasions)

    return {
        'episode_rewards': episode_rewards,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_steps': mean_steps,
        'mean_collisions': mean_collisions,
        'mean_lane_invasions': mean_lane_invasions
    }