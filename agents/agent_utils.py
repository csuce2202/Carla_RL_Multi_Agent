import os
import numpy as np
import torch
import gym
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


def evaluate_model(model, env, num_episodes=10, deterministic=True):
    """
    评估模型性能
    """
    episode_rewards = []
    episode_steps = []
    collision_counts = []
    lane_invasion_counts = []

    # 获取模型所在的设备
    device = next(model.policy.parameters()).device
    print(f"模型运行在设备: {device}")

    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        collision_count = 0
        lane_invasion_count = 0

        while not done:
            # 由于在特征提取器中已处理设备转换，可以直接使用原始观察值
            # 但为了保险起见，这里仍进行转换
            if isinstance(obs, dict):
                # 字典类型观察值不需要特殊处理，特征提取器会处理其中的张量
                pass
            else:
                # 单一类型观察值，如果是numpy数组则转换为tensor
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float()

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_step += 1

            # 记录碰撞和车道侵入
            if 'collision_count' in info:
                collision_count += info['collision_count']
            if 'lane_invasion_count' in info:
                lane_invasion_count += info['lane_invasion_count']

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        collision_counts.append(collision_count)
        lane_invasion_counts.append(lane_invasion_count)

        print(f"Episode {i + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {episode_step}")

    # 计算评估指标
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_steps = np.mean(episode_steps)
    mean_collisions = np.mean(collision_counts)
    mean_lane_invasions = np.mean(lane_invasion_counts)

    evaluation_results = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_steps': mean_steps,
        'mean_collisions': mean_collisions,
        'mean_lane_invasions': mean_lane_invasions,
        'episode_rewards': episode_rewards
    }

    return evaluation_results