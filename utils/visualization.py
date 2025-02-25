import os
import time
import math
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from pathlib import Path


def draw_lidar(surface, lidar_data, max_dist=50.0, dot_size=2, colormap='viridis', min_height=-2.0, max_height=5.0):
    """
    在pygame表面上绘制激光雷达点云

    参数:
        surface: pygame表面对象
        lidar_data: 激光雷达点云数据，形状为(N, 3)或(N, 4)
        max_dist: 最大显示距离
        dot_size: 点大小
        colormap: 颜色映射名称
        min_height: 最小高度
        max_height: 最大高度
    """
    width = surface.get_width()
    height = surface.get_height()
    center_x = width // 2
    center_y = height // 2
    scale = min(width, height) / (max_dist * 2)

    # 获取颜色映射
    colormap = cm.get_cmap(colormap)

    # 过滤距离过远的点
    if lidar_data.shape[1] >= 3:
        distances = np.sqrt(lidar_data[:, 0] ** 2 + lidar_data[:, 1] ** 2)
        valid_indices = distances < max_dist
        lidar_data = lidar_data[valid_indices]

    # 过滤高度超出范围的点
    if lidar_data.shape[1] >= 3:
        height_indices = (lidar_data[:, 2] >= min_height) & (lidar_data[:, 2] <= max_height)
        lidar_data = lidar_data[height_indices]

    # 绘制点云
    for point in lidar_data:
        # 点的位置
        x, y = point[0], point[1]

        # 转换为屏幕坐标
        screen_x = int(center_x + x * scale)
        screen_y = int(center_y - y * scale)  # 注意y轴方向相反

        # 检查是否在屏幕内
        if 0 <= screen_x < width and 0 <= screen_y < height:
            # 根据点的高度设置颜色
            if lidar_data.shape[1] >= 3:
                z = point[2]
                # 将高度归一化到[0, 1]范围
                normalized_height = max(0.0, min(1.0, (z - min_height) / (max_height - min_height)))
                color = colormap(normalized_height)
                color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            else:
                color = (0, 255, 0)  # 默认绿色

            # 绘制点
            pygame.draw.circle(surface, color, (screen_x, screen_y), dot_size)

    # 绘制参考网格
    grid_color = (100, 100, 100)
    for r in range(10, int(max_dist) + 1, 10):
        radius = int(r * scale)
        pygame.draw.circle(surface, grid_color, (center_x, center_y), radius, 1)

    # 绘制坐标轴
    axis_color = (200, 200, 200)
    pygame.draw.line(surface, axis_color, (center_x, 0), (center_x, height), 1)
    pygame.draw.line(surface, axis_color, (0, center_y), (width, center_y), 1)

    # 添加距离标签
    font = pygame.font.SysFont('Arial', 12)
    for r in range(10, int(max_dist) + 1, 10):
        radius = int(r * scale)
        label = font.render(f"{r}m", True, (200, 200, 200))
        surface.blit(label, (center_x + radius, center_y))


def draw_vehicle_info(surface, info, position=(10, 10), font_size=18):
    """
    在pygame表面上绘制车辆信息

    参数:
        surface: pygame表面对象
        info: 包含车辆信息的字典
        position: 起始位置
        font_size: 字体大小
    """
    font = pygame.font.SysFont('Arial', font_size)
    x, y = position
    line_height = font_size + 5

    # 绘制车速
    if 'speed' in info:
        speed_text = font.render(f"Speed: {info['speed']:.1f} km/h", True, (255, 255, 255))
        surface.blit(speed_text, (x, y))
        y += line_height

    # 绘制位置
    if 'location' in info:
        loc_x, loc_y, loc_z = info['location']
        loc_text = font.render(f"Position: ({loc_x:.1f}, {loc_y:.1f}, {loc_z:.1f})", True, (255, 255, 255))
        surface.blit(loc_text, (x, y))
        y += line_height

    # 绘制方向
    if 'rotation' in info:
        pitch, yaw, roll = info['rotation']
        rot_text = font.render(f"Rotation: ({pitch:.1f}, {yaw:.1f}, {roll:.1f})", True, (255, 255, 255))
        surface.blit(rot_text, (x, y))
        y += line_height

    # 绘制碰撞信息
    if 'collision_count' in info:
        collision_color = (255, 0, 0) if info['collision_count'] > 0 else (255, 255, 255)
        collision_text = font.render(f"Collision: {info['collision_count']}", True, collision_color)
        surface.blit(collision_text, (x, y))
        y += line_height

    # 绘制车道偏离信息
    if 'lane_invasion_count' in info:
        lane_color = (255, 165, 0) if info['lane_invasion_count'] > 0 else (255, 255, 255)
        lane_text = font.render(f"Lane Invasion: {info['lane_invasion_count']}", True, lane_color)
        surface.blit(lane_text, (x, y))
        y += line_height

    # 绘制奖励
    if 'reward' in info:
        reward_color = (0, 255, 0) if info['reward'] > 0 else (255, 0, 0)
        reward_text = font.render(f"Reward: {info['reward']:.2f}", True, reward_color)
        surface.blit(reward_text, (x, y))


def plot_to_surface(fig, size=(640, 480)):
    """
    将matplotlib图表转换为pygame表面

    参数:
        fig: matplotlib图表
        size: 目标表面大小

    返回:
        pygame.Surface: 包含图表的pygame表面
    """
    # 渲染图表到canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # 获取图表数据
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()

    # 创建pygame表面
    surf = pygame.image.fromstring(raw_data, size, "RGB")

    # 如果需要调整大小
    if size != surf.get_size():
        surf = pygame.transform.scale(surf, size)

    return surf


def plot_reward_history(rewards, size=(640, 240), window=100):
    """
    绘制奖励历史图表

    参数:
        rewards: 奖励历史列表
        size: 图表大小
        window: 移动平均窗口大小

    返回:
        pygame.Surface: 包含图表的pygame表面
    """
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)

    # 绘制原始奖励
    ax.plot(rewards, 'b-', alpha=0.3, label='Reward')

    # 计算移动平均
    if len(rewards) >= window:
        avg_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        avg_x = np.arange(window - 1, len(rewards))
        ax.plot(avg_x, avg_rewards, 'r-', label=f'{window}-ep Moving Avg')

    ax.set_title('Reward History')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # 自动调整布局
    plt.tight_layout()

    # 转换为pygame表面
    return plot_to_surface(fig, size)


def save_trajectory_plot(trajectory, filename, map_name=None):
    """
    保存轨迹图

    参数:
        trajectory: 轨迹点列表，每个点为(x, y)
        filename: 保存文件名
        map_name: 地图名称
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # 提取x和y坐标
    x = [point[0] for point in trajectory]
    y = [point[1] for point in trajectory]

    # 绘制轨迹
    ax.plot(x, y, 'b-', linewidth=2)
    ax.scatter(x[0], y[0], color='green', s=100, label='Start')
    ax.scatter(x[-1], y[-1], color='red', s=100, label='End')

    # 设置标题和标签
    title = f'Vehicle Trajectory'
    if map_name:
        title += f' on {map_name}'
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True)
    ax.legend()

    # 设置相等的轴比例
    ax.set_aspect('equal')

    # 保存图表
    plt.savefig(filename)
    plt.close(fig)


def create_heatmap(points, filename, map_name=None, resolution=1.0, max_dist=None):
    """
    创建位置热图

    参数:
        points: 位置点列表，每个点为(x, y)
        filename: 保存文件名
        map_name: 地图名称
        resolution: 网格分辨率
        max_dist: 最大距离限制
    """
    # 提取x和y坐标
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])

    # 确定数据范围
    if max_dist is None:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
    else:
        x_min, x_max = -max_dist, max_dist
        y_min, y_max = -max_dist, max_dist

    # 创建网格
    x_bins = np.arange(x_min, x_max + resolution, resolution)
    y_bins = np.arange(y_min, y_max + resolution, resolution)

    # 计算频率热图
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # 绘制热图
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap.T, cmap='viridis', origin='lower',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   interpolation='nearest')

    # 添加颜色条
    plt.colorbar(im, ax=ax, label='Frequency')

    # 设置标题和标签
    title = 'Vehicle Position Heatmap'
    if map_name:
        title += f' on {map_name}'
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # 保存图表
    plt.savefig(filename)
    plt.close(fig)