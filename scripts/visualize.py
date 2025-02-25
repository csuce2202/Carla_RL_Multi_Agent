import os
import time
import cv2
import numpy as np
import pygame
import torch
from pathlib import Path
from stable_baselines3 import PPO

from RL_multi.envs.carla_env import CARLAEnv
from RL_multi.utils.visualization import draw_lidar, draw_vehicle_info


def visualize_agents(env_config, agent_config, train_config, checkpoint_path):
    """
    可视化训练好的智能体

    参数:
        env_config: 环境配置
        agent_config: 智能体配置
        train_config: 训练配置
        checkpoint_path: 模型检查点路径
    """
    print(f"开始可视化模型: {checkpoint_path}")

    # 创建可视化环境
    print("创建可视化环境...")
    env_config_vis = env_config.copy()
    env_config_vis['render_mode'] = 'human'  # 使用人类可视化模式

    env = CARLAEnv(env_config_vis)

    # 设置随机种子
    seed = train_config.get('seed', 42)
    env.reset(seed=seed)

    # 加载模型 - 使用一致的设备策略
    print(f"加载模型: {checkpoint_path}")
    # 确定是否使用CUDA
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"使用设备: {device}")

    # 加载模型，直接指定设备
    model = PPO.load(checkpoint_path, device=device)

    # 确保策略在指定设备上
    model.policy = model.policy.to(device)

    # 显式设置所有子模块到正确设备
    for param in model.policy.parameters():
        param.data = param.data.to(device)

    # 初始化Pygame
    pygame.init()
    pygame.font.init()
    display_width = env_config_vis.get('camera_resolution_x', 640)
    display_height = env_config_vis.get('camera_resolution_y', 480) + 200  # 额外空间用于显示信息
    display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("CARLA RL Agent Visualization")
    clock = pygame.time.Clock()

    # 初始化录制
    record = train_config.get('record_video', False)
    record_dir = None
    video_writer = None

    if record:
        record_dir = os.path.join(os.path.dirname(checkpoint_path), 'visualization')
        os.makedirs(record_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(record_dir, f'carla_agent_{timestamp}.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (display_width, display_height))
        print(f"录制视频将保存到: {video_path}")

    # 可视化参数
    max_episodes = train_config.get('vis_episodes', 5)
    max_steps = env_config_vis.get('max_episode_steps', 1000)
    deterministic = train_config.get('vis_deterministic', True)

    try:
        for episode in range(max_episodes):
            print(f"\n===== 回合 {episode + 1}/{max_episodes} =====")
            obs, _ = env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done and step < max_steps:
                # 渲染当前帧
                render_img = env.render()

                # 统一处理观测数据到与模型相同的设备
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).to(device)
                elif isinstance(obs, dict):
                    obs_tensor = {}
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray):
                            obs_tensor[k] = torch.FloatTensor(v).to(device)
                        elif isinstance(v, torch.Tensor):
                            obs_tensor[k] = v.to(device)
                        else:
                            obs_tensor[k] = v
                else:
                    obs_tensor = obs  # 如果是其他类型，保持不变

                # 使用处理后的obs进行预测
                action, _ = model.predict(obs_tensor, deterministic=deterministic)

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                # 准备显示内容
                if render_img is not None:
                    # 将RGB图像转换为pygame表面
                    render_img = render_img.swapaxes(0, 1)
                    surface = pygame.surfarray.make_surface(render_img)

                    # 显示图像
                    display.fill((0, 0, 0))
                    display.blit(surface, (0, 0))

                    # 显示信息面板
                    info_surface = pygame.Surface((display_width, 200))
                    info_surface.fill((50, 50, 50))

                    # 添加文本信息
                    font = pygame.font.SysFont('Arial', 20)

                    # 回合信息
                    episode_text = font.render(f"Episode: {episode + 1}/{max_episodes}, Step: {step}", True,
                                               (255, 255, 255))
                    display.blit(episode_text, (10, display_height - 190))

                    # 奖励信息
                    reward_text = font.render(f"Reward: {reward:.2f}, Total: {total_reward:.2f}", True, (255, 255, 255))
                    display.blit(reward_text, (10, display_height - 160))

                    # 车辆状态
                    if 'speed' in info:
                        speed_text = font.render(f"Speed: {info['speed']:.2f} km/h", True, (255, 255, 255))
                        display.blit(speed_text, (10, display_height - 130))

                    # 碰撞信息
                    if 'collision_count' in info:
                        collision_color = (255, 0, 0) if info['collision_count'] > 0 else (255, 255, 255)
                        collision_text = font.render(f"Collision: {info['collision_count']}", True, collision_color)
                        display.blit(collision_text, (10, display_height - 100))

                    # 车道偏离信息
                    if 'lane_invasion_count' in info:
                        lane_color = (255, 165, 0) if info['lane_invasion_count'] > 0 else (255, 255, 255)
                        lane_text = font.render(f"Lane Invasion: {info['lane_invasion_count']}", True, lane_color)
                        display.blit(lane_text, (10, display_height - 70))

                    # 动作信息
                    if len(action) >= 2:
                        throttle_brake = float(action[0])
                        steer = float(action[1])

                        throttle = max(0, throttle_brake) if throttle_brake >= 0 else 0
                        brake = abs(min(0, throttle_brake)) if throttle_brake < 0 else 0

                        action_text = font.render(
                            f"Throttle: {throttle:.2f}, Brake: {brake:.2f}, Steer: {steer:.2f}",
                            True, (255, 255, 255)
                        )
                        display.blit(action_text, (10, display_height - 40))

                    # 更新显示
                    pygame.display.flip()

                    # 录制视频帧
                    if record and video_writer is not None:
                        # 将pygame表面转换为OpenCV图像
                        pygame_surface = pygame.surfarray.array3d(display)
                        # 转换颜色空间: RGB -> BGR
                        cv_image = pygame_surface.swapaxes(0, 1)
                        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                        video_writer.write(cv_image)

                # 控制帧率
                clock.tick(20)

                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            done = True
                            break

            print(f"回合 {episode + 1} 完成，总奖励: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\n用户中断可视化")

    finally:
        # 清理资源
        if record and video_writer is not None:
            video_writer.release()

        pygame.quit()
        env.close()
        print("可视化结束")


def visualize_with_sensors(env_config, agent_config, train_config, checkpoint_path):
    """
    使用多个传感器视图可视化智能体

    参数:
        env_config: 环境配置
        agent_config: 智能体配置
        train_config: 训练配置
        checkpoint_path: 模型检查点路径
    """
    print(f"开始多传感器可视化: {checkpoint_path}")

    # 创建可视化环境
    print("创建可视化环境...")
    env_config_vis = env_config.copy()
    env_config_vis['render_mode'] = 'rgb_array'  # 使用RGB数组模式以获取原始图像数据

    env = CARLAEnv(env_config_vis)

    # 设置随机种子
    seed = train_config.get('seed', 42)
    env.reset(seed=seed)

    # 加载模型 - 使用一致的设备策略
    print(f"加载模型: {checkpoint_path}")
    # 确定是否使用CUDA
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"使用设备: {device}")

    # 加载模型，直接指定设备
    model = PPO.load(checkpoint_path, device=device)

    # 确保策略在指定设备上
    model.policy = model.policy.to(device)

    # 显式设置所有子模块到正确设备
    for param in model.policy.parameters():
        param.data = param.data.to(device)

    # 初始化Pygame
    pygame.init()
    pygame.font.init()

    # 设置多视图显示
    view_width = env_config_vis.get('camera_resolution_x', 640)
    view_height = env_config_vis.get('camera_resolution_y', 480)

    # 创建主窗口：左侧为相机视图，右侧为激光雷达和其他传感器数据
    display_width = view_width * 2
    display_height = view_height + 200  # 额外空间用于显示信息
    display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("CARLA RL Agent Multi-Sensor Visualization")
    clock = pygame.time.Clock()

    # 初始化录制
    record = train_config.get('record_video', False)
    record_dir = None
    video_writer = None

    if record:
        record_dir = os.path.join(os.path.dirname(checkpoint_path), 'visualization')
        os.makedirs(record_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = os.path.join(record_dir, f'carla_agent_multisensor_{timestamp}.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (display_width, display_height))
        print(f"录制视频将保存到: {video_path}")

    # 可视化参数
    max_episodes = train_config.get('vis_episodes', 5)
    max_steps = env_config_vis.get('max_episode_steps', 1000)
    deterministic = train_config.get('vis_deterministic', True)

    try:
        for episode in range(max_episodes):
            print(f"\n===== 回合 {episode + 1}/{max_episodes} =====")
            obs, _ = env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done and step < max_steps:
                # 统一处理观测数据到与模型相同的设备
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).to(device)
                elif isinstance(obs, dict):
                    obs_tensor = {}
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray):
                            obs_tensor[k] = torch.FloatTensor(v).to(device)
                        elif isinstance(v, torch.Tensor):
                            obs_tensor[k] = v.to(device)
                        else:
                            obs_tensor[k] = v
                else:
                    obs_tensor = obs  # 如果是其他类型，保持不变

                # 使用处理后的obs进行预测
                action, _ = model.predict(obs_tensor, deterministic=deterministic)

                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                # 获取传感器数据
                camera_img = None
                lidar_data = None

                if isinstance(obs, dict):
                    if 'camera' in obs:
                        camera_img = obs['camera']
                    if 'lidar' in obs:
                        lidar_data = obs['lidar']
                elif hasattr(env, 'sensor_data') and 0 in env.sensor_data:
                    if 'camera' in env.sensor_data[0]:
                        camera_img = env.sensor_data[0]['camera']
                    if 'lidar' in env.sensor_data[0]:
                        lidar_data = env.sensor_data[0]['lidar']

                # 准备显示内容
                display.fill((0, 0, 0))

                # 显示相机图像
                if camera_img is not None:
                    # 将RGB图像转换为pygame表面
                    camera_surface = pygame.surfarray.make_surface(camera_img.swapaxes(0, 1))
                    display.blit(camera_surface, (0, 0))

                # 显示激光雷达点云
                if lidar_data is not None:
                    lidar_surface = pygame.Surface((view_width, view_height))
                    lidar_surface.fill((0, 0, 0))

                    # 绘制点云
                    draw_lidar(lidar_surface, lidar_data, max_dist=50.0)

                    display.blit(lidar_surface, (view_width, 0))

                # 显示信息面板
                info_surface = pygame.Surface((display_width, 200))
                info_surface.fill((50, 50, 50))
                display.blit(info_surface, (0, view_height))

                # 添加文本信息
                font = pygame.font.SysFont('Arial', 20)

                # 回合信息
                episode_text = font.render(f"Episode: {episode + 1}/{max_episodes}, Step: {step}", True,
                                           (255, 255, 255))
                display.blit(episode_text, (10, view_height + 10))

                # 奖励信息
                reward_text = font.render(f"Reward: {reward:.2f}, Total: {total_reward:.2f}", True, (255, 255, 255))
                display.blit(reward_text, (10, view_height + 40))

                # 车辆状态
                if 'speed' in info:
                    speed_text = font.render(f"Speed: {info['speed']:.2f} km/h", True, (255, 255, 255))
                    display.blit(speed_text, (10, view_height + 70))

                # 碰撞信息
                if 'collision_count' in info:
                    collision_color = (255, 0, 0) if info['collision_count'] > 0 else (255, 255, 255)
                    collision_text = font.render(f"Collision: {info['collision_count']}", True, collision_color)
                    display.blit(collision_text, (10, view_height + 100))

                # 车道偏离信息
                if 'lane_invasion_count' in info:
                    lane_color = (255, 165, 0) if info['lane_invasion_count'] > 0 else (255, 255, 255)
                    lane_text = font.render(f"Lane Invasion: {info['lane_invasion_count']}", True, lane_color)
                    display.blit(lane_text, (10, view_height + 130))

                # 动作信息
                if len(action) >= 2:
                    throttle_brake = float(action[0])
                    steer = float(action[1])

                    throttle = max(0, throttle_brake) if throttle_brake >= 0 else 0
                    brake = abs(min(0, throttle_brake)) if throttle_brake < 0 else 0

                    action_text = font.render(
                        f"Throttle: {throttle:.2f}, Brake: {brake:.2f}, Steer: {steer:.2f}",
                        True, (255, 255, 255)
                    )
                    display.blit(action_text, (10, view_height + 160))

                    # 位置信息
                if 'location' in info:
                    loc_x, loc_y, loc_z = info['location']
                    location_text = font.render(f"Position: ({loc_x:.1f}, {loc_y:.1f}, {loc_z:.1f})", True,
                                                (255, 255, 255))
                    display.blit(location_text, (display_width // 2 + 10, view_height + 10))

                    # 更新显示
                pygame.display.flip()

                # 录制视频帧
                if record and video_writer is not None:
                    # 将pygame表面转换为OpenCV图像
                    pygame_surface = pygame.surfarray.array3d(display)
                    # 转换颜色空间: RGB -> BGR
                    cv_image = pygame_surface.swapaxes(0, 1)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    video_writer.write(cv_image)

                # 控制帧率
                clock.tick(20)

                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            done = True
                            break

            print(f"回合 {episode + 1} 完成，总奖励: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\n用户中断可视化")

    finally:
        # 清理资源
        if record and video_writer is not None:
            video_writer.release()

        pygame.quit()
        env.close()
        print("可视化结束")


if __name__ == "__main__":
    # 测试可视化函数
    from RL_multi.configs.env_config import ENV_CONFIG
    from RL_multi.configs.agent_config import AGENT_CONFIG
    from RL_multi.configs.train_config import TRAIN_CONFIG

    # 假设的模型路径
    checkpoint_path = 'D:/Research/Nick Yu/ADV/RL_multi/checkpoints/carla_rl_20250225-200501/final_model.zip'

    # 确保路径存在，否则跳过实际可视化
    if os.path.exists(checkpoint_path):
        visualize_agents(ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG, checkpoint_path)

        # 多传感器可视化
        visualize_with_sensors(ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG, checkpoint_path)
    else:
        print(f"模型文件不存在: {checkpoint_path}")
        print("跳过可视化")