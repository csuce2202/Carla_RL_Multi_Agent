import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class CarlaFeatureExtractor(BaseFeaturesExtractor):
    """
    定制化的特征提取器，用于处理CARLA环境的复杂观察数据
    """

    def __init__(self, observation_space, features_dim=256):
        super(CarlaFeatureExtractor, self).__init__(observation_space, features_dim)

        # 显式保存observation_space，修复"'CarlaFeatureExtractor' object has no attribute 'observation_space'"错误
        self.observation_space = observation_space

        # 获取设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 检查是否为多输入空间
        if isinstance(observation_space, spaces.Dict):
            # 创建各个特征提取子网络
            self.extractors = {}

            # 图像提取器
            if 'camera' in observation_space.spaces:
                camera_space = observation_space.spaces['camera']
                n_channels = camera_space.shape[2]

                # 打印输入形状以进行调试
                print(f"Camera input shape: {camera_space.shape}")

                # 根据输入大小调整卷积层参数
                # 使用更小的卷积核和步长
                self.extractors['camera'] = nn.Sequential(
                    nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),  # 减小卷积核并增加padding
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),  # 使用池化层代替大步长卷积
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Flatten()
                ).to(self.device)

                # 计算CNN输出维度
                with torch.no_grad():
                    # 打印原始尺寸进行调试
                    print(f"Creating dummy image with shape: {(1,) + camera_space.shape}")
                    dummy_img = torch.zeros((1,) + camera_space.shape, device=self.device)
                    dummy_img = dummy_img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    print(f"Permuted dummy image shape: {dummy_img.shape}")
                    # 进行前向计算来获取输出维度
                    try:
                        camera_features = self.extractors['camera'](dummy_img)
                        camera_out_dim = camera_features.shape[1]
                        print(f"CNN output dimension: {camera_out_dim}")
                    except Exception as e:
                        print(f"Error during CNN dimension calculation: {e}")
                        # 如果CNN前向传播失败，使用一个合理的默认值
                        camera_out_dim = 256

                self.camera_linear = nn.Sequential(
                    nn.Linear(camera_out_dim, 256),
                    nn.ReLU()
                ).to(self.device)

            # 激光雷达提取器
            if 'lidar' in observation_space.spaces:
                lidar_space = observation_space.spaces['lidar']
                lidar_dims = np.prod(lidar_space.shape)
                print(f"Lidar input dimension: {lidar_dims}")

                self.extractors['lidar'] = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(lidar_dims, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                ).to(self.device)

            # 其他传感器数据提取器
            for key, subspace in observation_space.spaces.items():
                if key not in ['camera', 'lidar']:
                    dims = np.prod(subspace.shape)
                    print(f"{key} input dimension: {dims}")
                    self.extractors[key] = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(dims, 32),
                        nn.ReLU()
                    ).to(self.device)

            # 计算总特征维度
            self.feature_dim = 0
            if 'camera' in self.extractors:
                self.feature_dim += 256
            if 'lidar' in self.extractors:
                self.feature_dim += 128

            for key in self.extractors:
                if key not in ['camera', 'lidar']:
                    self.feature_dim += 32

            # 最终组合层
            self.final_layer = nn.Sequential(
                nn.Linear(self.feature_dim, features_dim),
                nn.ReLU()
            ).to(self.device)

        else:
            # 单一输入空间（通常是图像）
            if len(observation_space.shape) == 3:
                # 图像输入
                n_input_channels = observation_space.shape[2]

                # 打印输入形状以进行调试
                print(f"Image input shape: {observation_space.shape}")

                # 调整卷积层参数
                self.cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Flatten()
                ).to(self.device)

                # 计算CNN输出维度
                with torch.no_grad():
                    dummy_img = torch.zeros((1,) + observation_space.shape, device=self.device)
                    dummy_img = dummy_img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    print(f"Permuted dummy image shape: {dummy_img.shape}")
                    try:
                        features = self.cnn(dummy_img)
                        n_flatten = features.shape[1]
                        print(f"CNN output dimension: {n_flatten}")
                    except Exception as e:
                        print(f"Error during CNN dimension calculation: {e}")
                        # 如果CNN前向传播失败，使用一个合理的默认值
                        n_flatten = 256

                self.linear = nn.Sequential(
                    nn.Linear(n_flatten, features_dim),
                    nn.ReLU()
                ).to(self.device)
            else:
                # 其他类型输入
                n_input = np.prod(observation_space.shape)
                print(f"Other input dimension: {n_input}")

                self.linear = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(n_input, 256),
                    nn.ReLU(),
                    nn.Linear(256, features_dim),
                    nn.ReLU()
                ).to(self.device)

    def forward(self, observations):
        """前向传播函数"""
        # 获取当前模型的设备
        device = next(self.parameters()).device

        if isinstance(self.observation_space, spaces.Dict):
            # 多输入处理
            extracted_features = []

            if 'camera' in self.extractors:
                # 确保camera数据在正确设备上
                if 'camera' in observations:
                    camera = observations['camera']
                    # 检查并转换张量设备
                    if isinstance(camera, np.ndarray):
                        camera = torch.from_numpy(camera).float().to(device)
                    elif isinstance(camera, torch.Tensor):
                        camera = camera.float().to(device)

                    camera = camera.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                    camera_features = self.camera_linear(self.extractors['camera'](camera))
                    extracted_features.append(camera_features)

            # 处理其他传感器数据
            for key, extractor in self.extractors.items():
                if key != 'camera' and key in observations:
                    data = observations[key]

                    # 确保数据在正确设备上
                    if isinstance(data, np.ndarray):
                        data = torch.from_numpy(data).float().to(device)
                    elif isinstance(data, torch.Tensor):
                        data = data.float().to(device)

                    # 处理lidar数据的特殊情况
                    if key == 'lidar' and len(data.shape) > 2:
                        features = self.extractors['lidar'](data)
                    else:
                        features = extractor(data)

                    extracted_features.append(features)

            # 确保有提取的特征
            if not extracted_features:
                raise ValueError("没有从观察空间中提取到任何特征")

            # 组合所有特征
            combined_features = torch.cat(extracted_features, dim=1)

            # 返回最终特征
            return self.final_layer(combined_features)
        else:
            # 单一输入处理
            # 确保数据在正确设备上
            if isinstance(observations, np.ndarray):
                observations = torch.from_numpy(observations).float().to(device)
            elif isinstance(observations, torch.Tensor):
                observations = observations.float().to(device)

            if hasattr(self, 'cnn'):
                # 图像处理
                observations = observations.permute(0, 3, 1, 2)
                features = self.cnn(observations)
                return self.linear(features)
            else:
                # 其他类型处理
                return self.linear(observations)


def create_ppo_model(env, config, seed=None, device='auto'):
    """创建PPO模型"""
    # 使用正确的设备
    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"创建模型使用设备: {device}")

    # 策略网络参数
    policy_kwargs = {
        'features_extractor_class': CarlaFeatureExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
        'net_arch': dict(pi=[256, 256], vf=[256, 256])  # 修改为字典格式，不是列表中的字典
    }

    # 创建模型
    model = PPO(
        policy="MultiInputPolicy" if isinstance(env.observation_space, spaces.Dict) else "CnnPolicy",
        env=env,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 64),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        clip_range_vf=config.get('clip_range_vf', None),
        normalize_advantage=config.get('normalize_advantage', True),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        use_sde=config.get('use_sde', False),
        sde_sample_freq=config.get('sde_sample_freq', -1),
        target_kl=config.get('target_kl', None),
        tensorboard_log=config.get('tensorboard_log', "./tensorboard_logs/"),
        policy_kwargs=policy_kwargs,
        verbose=config.get('verbose', 1),
        seed=seed,
        device=device
    )

    return model


def load_ppo_model(env, path, device='auto'):
    """加载PPO模型"""
    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return PPO.load(path, env=env, device=device)


def save_ppo_model(model, path):
    """保存PPO模型"""
    model.save(path)