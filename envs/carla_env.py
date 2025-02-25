import os
import sys
import time
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import carla
import pygame
import queue
import cv2
import math
import weakref
from collections import deque


class CARLAEnv(gym.Env):
    """
    CARLA 环境包装器，支持多智能体强化学习
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, config):
        super(CARLAEnv, self).__init__()

        self.config = config
        self.render_mode = config.get('render_mode', 'rgb_array')
        self.num_agents = config.get('num_agents', 1)
        self.synchronous = config.get('synchronous', True)
        self.delta_seconds = config.get('delta_seconds', 0.05)

        # 用于多帧堆叠 - 移到前面，在_setup_spaces()之前
        self.stacked_frames = config.get('stacked_frames', 4)

        # 连接CARLA服务器
        self.client = None
        self.world = None
        self.map = None
        self.spawn_points = []  # Initialize as empty list
        self.connect_carla()

        # 初始化输入/输出大小
        self._setup_spaces()

        # 初始化车辆和传感器
        self.vehicles = []
        self.sensors = {}
        self.sensor_data = {}
        self.collision_history = {}
        self.lane_invasion_history = {}

        # 可视化相关
        self.pygame_display = None
        self.pygame_clock = None
        if self.render_mode == 'human':
            self._setup_pygame()

        # 帧堆叠数据结构
        self.frame_stacks = {}

        # 存储最后一次的奖励值，用于记录调试
        self.last_rewards = [0.0] * self.num_agents

        # 记录当前步数
        self.current_step = 0
        self.max_episode_steps = config.get('max_episode_steps', 1000)

    def connect_carla(self):
        """连接到CARLA服务器"""
        carla_port = self.config.get('carla_port', 2000)
        carla_timeout = self.config.get('carla_timeout', 20.0)

        try:
            self.client = carla.Client('localhost', carla_port)
            self.client.set_timeout(carla_timeout)

            # 设置CARLA世界和地图
            self.world = self.client.get_world()
            current_map = self.world.get_map().name
            requested_map = self.config.get('map', 'Town01')

            if current_map != requested_map:
                print(f"Changing map from {current_map} to {requested_map}")
                self.world = self.client.load_world(requested_map)

            self.map = self.world.get_map()
            self.spawn_points = self.map.get_spawn_points()
            print(f"Number of spawn points: {len(self.spawn_points)}")  # Debug statement

            # 设置同步模式
            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.delta_seconds
                settings.no_rendering_mode = self.config.get('no_rendering', False)
                self.world.apply_settings(settings)

            # 设置天气
            weather_preset = self.config.get('weather', 'ClearNoon')
            self._set_weather(weather_preset)

            # 清除现有车辆和行人
            self._clear_actors()

            # 简单的检查，确保连接成功
            if not self.world:
                raise ConnectionError("无法连接到CARLA世界")

            # Tick world to ensure settings are applied
            if self.synchronous:
                self.world.tick()

            # 重新获取spawn points to ensure they're up to date
            self.spawn_points = self.world.get_map().get_spawn_points()
            print(f"Updated number of spawn points: {len(self.spawn_points)}")

        except Exception as e:
            print(f"连接CARLA服务器失败: {e}")
            sys.exit(1)

    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        # 观察空间组成部分
        observation_parts = {}

        # 相机观察空间
        if self.config.get('observation_space', {}).get('camera', True):
            camera_height = self.config.get('camera_resolution_y', 480)
            camera_width = self.config.get('camera_resolution_x', 640)
            observation_parts['camera'] = spaces.Box(
                low=0, high=255,
                shape=(camera_height, camera_width, 3 * self.stacked_frames),
                dtype=np.uint8
            )

        # LiDAR观察空间
        if self.config.get('observation_space', {}).get('lidar', True):
            lidar_channels = 32  # 激光雷达点云数据的通道数
            lidar_points = 1024  # 每个通道的点数
            observation_parts['lidar'] = spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(lidar_channels, lidar_points, 3),  # x, y, z
                dtype=np.float32
            )

        # 雷达观察空间
        if self.config.get('observation_space', {}).get('radar', False):
            radar_points = 256
            observation_parts['radar'] = spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(radar_points, 4),  # x, y, z, velocity
                dtype=np.float32
            )

        # GNSS观察空间
        if self.config.get('observation_space', {}).get('gnss', True):
            observation_parts['gnss'] = spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(3,),  # x, y, z
                dtype=np.float32
            )

        # IMU观察空间
        if self.config.get('observation_space', {}).get('imu', True):
            observation_parts['imu'] = spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(7,),  # 加速度(3) + 陀螺仪(3) + 指南针(1)
                dtype=np.float32
            )

        # 车速表观察空间
        if self.config.get('observation_space', {}).get('speedometer', True):
            observation_parts['speedometer'] = spaces.Box(
                low=0, high=float('inf'),
                shape=(1,),
                dtype=np.float32
            )

        # 车道侵入观察空间
        if self.config.get('observation_space', {}).get('lane_invasion', True):
            observation_parts['lane_invasion'] = spaces.Box(
                low=0, high=1,
                shape=(1,),
                dtype=np.float32
            )

        # 碰撞观察空间
        if self.config.get('observation_space', {}).get('collision', True):
            observation_parts['collision'] = spaces.Box(
                low=0, high=float('inf'),
                shape=(1,),
                dtype=np.float32
            )

        # 组合观察空间
        if len(observation_parts) == 1:
            # 只有一个观察空间，直接使用它
            self.observation_space = next(iter(observation_parts.values()))
        else:
            # 多个观察空间，使用Dict
            self.observation_space = spaces.Dict(observation_parts)

        # 设置动作空间
        action_type = self.config.get('action_space', 'continuous')
        if action_type == 'continuous':
            # 连续动作空间: 油门/刹车(-1到1), 转向(-1到1)
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        else:  # discrete
            # 离散动作空间: 9种组合(3种油门/刹车 x 3种转向)
            self.action_space = spaces.Discrete(9)

    def _set_weather(self, preset):
        """设置CARLA世界的天气预设"""
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'CloudyNoon': carla.WeatherParameters.CloudyNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
            'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
            'CloudySunset': carla.WeatherParameters.CloudySunset,
            'WetSunset': carla.WeatherParameters.WetSunset,
            'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,
            'MidRainSunset': carla.WeatherParameters.MidRainSunset,
            'HardRainSunset': carla.WeatherParameters.HardRainSunset,
            'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
        }

        if preset in weather_presets:
            self.world.set_weather(weather_presets[preset])
        else:
            print(f"未知的天气预设: {preset}，使用默认的ClearNoon")
            self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _clear_actors(self):
        """清除世界中的所有车辆和行人"""
        for actor in self.world.get_actors():
            if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
                actor.destroy()

    def _clear_all_actors(self):
        """清除所有车辆和传感器"""
        # 首先清除传感器
        for agent_id in self.sensors:
            for sensor_type in self.sensors[agent_id]:
                if self.sensors[agent_id][sensor_type] is not None:
                    self.sensors[agent_id][sensor_type].destroy()

        # 然后清除车辆
        for vehicle in self.vehicles:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()

    def _setup_pygame(self):
        """初始化Pygame用于可视化"""
        if not pygame.get_init():
            pygame.init()
        pygame.font.init()

        width = self.config.get('camera_resolution_x', 640)
        height = self.config.get('camera_resolution_y', 480)

        self.pygame_display = pygame.display.set_mode(
            (width, height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.pygame_clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        """
        super().reset(seed=seed)

        # 清除现有车辆和传感器
        self._clear_all_actors()

        # 初始化数据结构
        self.vehicles = []
        self.sensors = {i: {} for i in range(self.num_agents)}
        self.sensor_data = {i: {} for i in range(self.num_agents)}
        self.collision_history = {i: 0.0 for i in range(self.num_agents)}
        self.lane_invasion_history = {i: 0.0 for i in range(self.num_agents)}
        self.frame_stacks = {i: {} for i in range(self.num_agents)}

        # 重置步数
        self.current_step = 0

        # Ensure the map and world are still valid
        if self.world is None or self.map is None:
            print("Reconnecting to CARLA server...")
            self.connect_carla()

        # Refresh spawn points
        self.spawn_points = self.world.get_map().get_spawn_points()
        print(f"Number of spawn points at reset: {len(self.spawn_points)}")

        # 为每个智能体生成车辆
        if len(self.spawn_points) < self.num_agents:
            raise ValueError(f"地图上没有足够的生成点，需要 {self.num_agents} 个，但只有 {len(self.spawn_points)} 个")

        # 随机选择不重复的生成点
        selected_spawns = random.sample(self.spawn_points, self.num_agents)

        # 为每个智能体创建车辆和传感器
        for i in range(self.num_agents):
            # 创建车辆
            blueprint = random.choice([bp for bp in self.world.get_blueprint_library().filter('vehicle.*')
                                       if int(bp.get_attribute('number_of_wheels')) == 4])

            # Try to spawn the vehicle, with retries
            vehicle = None
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    spawn_point = selected_spawns[i]
                    vehicle = self.world.spawn_actor(blueprint, spawn_point)
                    vehicle.set_autopilot(False)
                    self.vehicles.append(vehicle)
                    break
                except Exception as e:
                    print(f"Spawn attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt == max_retries - 1:
                        print("Failed to spawn vehicle after maximum retries")
                        raise
                    # Try a different spawn point
                    if len(self.spawn_points) > i + attempt + 1:
                        selected_spawns[i] = self.spawn_points[i + attempt + 1]

            # 为车辆添加传感器
            self._setup_sensors(i, vehicle)

        # 在同步模式下等待数据
        if self.synchronous:
            for _ in range(10):  # 给传感器一些时间来初始化
                self.world.tick()
                time.sleep(0.1)

        # 获取初始观察
        observations = {}
        for i in range(self.num_agents):
            observations[i] = self._get_observation(i)

        # 如果只有一个智能体，直接返回其观察而不是字典
        if self.num_agents == 1:
            return observations[0], {}

        return observations, {}

    def _setup_sensors(self, agent_id, vehicle):
        """为指定的车辆设置传感器"""
        # 创建相机
        if self.config.get('observation_space', {}).get('camera', True):
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.config.get('camera_resolution_x', 640)))
            camera_bp.set_attribute('image_size_y', str(self.config.get('camera_resolution_y', 480)))
            camera_bp.set_attribute('fov', '110')

            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

            # 设置图像回调
            camera.listen(lambda image: self._process_camera_data(agent_id, image))
            self.sensors[agent_id]['camera'] = camera

            # 初始化帧堆叠
            self.frame_stacks[agent_id]['camera'] = deque(maxlen=self.stacked_frames)

        # 创建激光雷达
        if self.config.get('observation_space', {}).get('lidar', True):
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('points_per_second', str(self.config.get('lidar_points_per_second', 100000)))
            lidar_bp.set_attribute('range', str(self.config.get('lidar_range', 100.0)))
            lidar_bp.set_attribute('rotation_frequency', '10')
            lidar_bp.set_attribute('channels', '32')

            lidar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

            # 设置点云回调
            lidar.listen(lambda data: self._process_lidar_data(agent_id, data))
            self.sensors[agent_id]['lidar'] = lidar

        # 创建雷达
        if self.config.get('observation_space', {}).get('radar', False):
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov', '30')
            radar_bp.set_attribute('vertical_fov', '30')
            radar_bp.set_attribute('points_per_second', str(self.config.get('radar_points_per_second', 1500)))
            radar_bp.set_attribute('range', str(self.config.get('radar_range', 100.0)))

            radar_transform = carla.Transform(carla.Location(x=1.5, z=1.0))
            radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)

            # 设置雷达回调
            radar.listen(lambda data: self._process_radar_data(agent_id, data))
            self.sensors[agent_id]['radar'] = radar

        # 创建GNSS
        if self.config.get('observation_space', {}).get('gnss', True):
            gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
            gnss_transform = carla.Transform(carla.Location(x=1.0, z=2.8))
            gnss = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)

            # 设置GNSS回调
            gnss.listen(lambda data: self._process_gnss_data(agent_id, data))
            self.sensors[agent_id]['gnss'] = gnss

        # 创建IMU
        if self.config.get('observation_space', {}).get('imu', True):
            imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
            imu_transform = carla.Transform(carla.Location(x=1.0, z=2.8))
            imu = self.world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)

            # 设置IMU回调
            imu.listen(lambda data: self._process_imu_data(agent_id, data))
            self.sensors[agent_id]['imu'] = imu

        # 创建碰撞传感器
        if self.config.get('observation_space', {}).get('collision', True):
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            collision_transform = carla.Transform()
            collision = self.world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle)

            # 设置碰撞回调
            collision.listen(lambda event: self._process_collision_data(agent_id, event))
            self.sensors[agent_id]['collision'] = collision

        # 创建车道侵入传感器
        if self.config.get('observation_space', {}).get('lane_invasion', True):
            lane_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            lane_transform = carla.Transform()
            lane = self.world.spawn_actor(lane_bp, lane_transform, attach_to=vehicle)

            # 设置车道侵入回调
            lane.listen(lambda event: self._process_lane_invasion_data(agent_id, event))
            self.sensors[agent_id]['lane'] = lane

    def _process_camera_data(self, agent_id, image):
        """处理相机数据"""
        # 将CARLA图像转换为Numpy数组
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 去掉alpha通道
        array = array[:, :, ::-1]  # BGR -> RGB

        # 存储数据
        self.sensor_data[agent_id]['camera'] = array

        # 更新帧堆叠
        if 'camera' in self.frame_stacks[agent_id]:
            self.frame_stacks[agent_id]['camera'].append(array)

    def _process_lidar_data(self, agent_id, data):
        """处理激光雷达数据"""
        # 转换为Numpy数组
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        # 将点云转换为3D坐标系
        lidar_data = np.array(points[:, :3])

        # 存储数据
        self.sensor_data[agent_id]['lidar'] = lidar_data

    def _process_radar_data(self, agent_id, data):
        """处理雷达数据"""
        # 转换为Numpy数组
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        # 存储数据
        self.sensor_data[agent_id]['radar'] = points

    def _process_gnss_data(self, agent_id, data):
        """处理GNSS数据"""
        # 存储GNSS坐标
        self.sensor_data[agent_id]['gnss'] = np.array([data.latitude, data.longitude, data.altitude], dtype=np.float32)

    def _process_collision_data(self, agent_id, event):
        """处理碰撞事件"""
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

        # 累积碰撞强度
        self.collision_history[agent_id] += intensity

        # 存储数据
        self.sensor_data[agent_id]['collision'] = np.array([self.collision_history[agent_id]], dtype=np.float32)

    def _process_lane_invasion_data(self, agent_id, event):
        """处理车道侵入事件"""
        # 增加车道侵入计数
        self.lane_invasion_history[agent_id] += 1

        # 存储数据
        self.sensor_data[agent_id]['lane_invasion'] = np.array([1.0], dtype=np.float32)

    def step(self, actions):
        """
        执行动作并返回新的观察、奖励、终止标志和信息

        参数:
            actions: 如果num_agents=1，则为单个动作；否则为动作字典 {agent_id: action}

        返回:
            observations: 如果num_agents=1，则为单个观察；否则为观察字典 {agent_id: observation}
            rewards: 如果num_agents=1，则为单个奖励；否则为奖励字典 {agent_id: reward}
            terminated: 如果num_agents=1，则为单个终止标志；否则为终止标志字典 {agent_id: terminated}
            truncated: 如果num_agents=1，则为单个截断标志；否则为截断标志字典 {agent_id: truncated}
            info: 附加信息字典
        """
        # 增加步数
        self.current_step += 1

        # 转换为动作字典
        action_dict = actions if isinstance(actions, dict) else {0: actions}

        # 应用动作到每个车辆
        for agent_id, action in action_dict.items():
            if agent_id < len(self.vehicles) and self.vehicles[agent_id].is_alive:
                self._apply_action(agent_id, action)

        # 在同步模式下推进世界
        if self.synchronous:
            self.world.tick()

        # 收集观察、奖励和终止标志
        observations = {}
        rewards = {}
        terminated = {}
        truncated = {}
        infos = {}

        for i in range(self.num_agents):
            # 只处理有效的智能体
            if i < len(self.vehicles) and self.vehicles[i].is_alive:
                # 获取观察
                observations[i] = self._get_observation(i)

                # 计算奖励
                rewards[i] = self._compute_reward(i)
                self.last_rewards[i] = rewards[i]

                # 检查终止条件
                terminated[i] = self._check_terminated(i)

                # 检查截断条件
                truncated[i] = self._check_truncated(i)

                # 收集信息
                infos[i] = self._get_info(i)
            else:
                # 无效智能体
                observations[i] = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype) \
                    if not isinstance(self.observation_space, spaces.Dict) \
                    else {k: np.zeros(v.shape, dtype=v.dtype)
                          for k, v in self.observation_space.spaces.items()}
                rewards[i] = 0.0
                terminated[i] = True
                truncated[i] = True
                infos[i] = {}

        # 简化单智能体情况的返回值
        if self.num_agents == 1:
            return observations[0], rewards[0], terminated[0], truncated[0], infos[0]

        return observations, rewards, terminated, truncated, infos

    def _apply_action(self, agent_id, action):
        """应用动作到指定的车辆"""
        vehicle = self.vehicles[agent_id]

        # 获取车辆控制
        control = vehicle.get_control()

        # 如果是连续动作空间
        if isinstance(self.action_space, spaces.Box):
            # action[0]: 油门(-1到0)/刹车(0到1)
            # action[1]: 转向(-1到1)
            throttle_brake = float(action[0])
            steer = float(action[1])

            # 应用动作平滑
            action_smoothing = self.config.get('action_smoothing', 0.9)
            control.throttle = max(0.0, throttle_brake) if throttle_brake >= 0.0 else 0.0
            control.brake = abs(min(0.0, throttle_brake)) if throttle_brake < 0.0 else 0.0
            control.steer = action_smoothing * control.steer + (1.0 - action_smoothing) * steer

        # 如果是离散动作空间
        else:
            # 将离散动作转换为连续控制
            # 0-8的动作映射到3x3矩阵(油门/刹车x转向)
            throttle_actions = [0.0, 0.5, 1.0]  # 无油门，中等油门，全油门
            brake_actions = [0.0, 0.5, 1.0]  # 无刹车，中等刹车，全刹车
            steer_actions = [-0.5, 0.0, 0.5]  # 左转，直行，右转

            action_id = int(action)

            # 计算油门/刹车和转向指数
            throttle_idx = action_id // 3
            steer_idx = action_id % 3

            if throttle_idx < 3:  # 油门动作
                control.throttle = throttle_actions[throttle_idx]
                control.brake = 0.0
            else:  # 刹车动作
                control.throttle = 0.0
                control.brake = brake_actions[throttle_idx - 3]

            control.steer = steer_actions[steer_idx]

        # 设置其他控制选项
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False

        # 应用控制到车辆
        vehicle.apply_control(control)

    def _get_observation(self, agent_id):
        """获取指定智能体的观察"""
        observation = {}

        # 检查是否有所有必需的传感器数据
        if not all(k in self.sensor_data[agent_id] for k in self.observation_space.spaces.keys()
                   if k not in ['speedometer', 'lane_invasion']):
            # 如果缺少必要数据，返回零观察
            return {k: np.zeros(v.shape, dtype=v.dtype)
                    for k, v in self.observation_space.spaces.items()}

        # 处理相机图像
        if 'camera' in self.observation_space.spaces:
            if len(self.frame_stacks[agent_id]['camera']) < self.stacked_frames:
                # 如果帧堆叠不足，用当前帧填充
                current_frame = self.sensor_data[agent_id]['camera']
                for _ in range(self.stacked_frames - len(self.frame_stacks[agent_id]['camera'])):
                    self.frame_stacks[agent_id]['camera'].append(current_frame)

            # 堆叠帧
            stacked_frames = np.concatenate([frame for frame in self.frame_stacks[agent_id]['camera']], axis=2)
            observation['camera'] = stacked_frames

        # 处理其他传感器数据
        for k in self.observation_space.spaces:
            if k != 'camera':
                if k == 'speedometer':
                    # 获取车速
                    velocity = self.vehicles[agent_id].get_velocity()
                    speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                    observation[k] = np.array([speed], dtype=np.float32)
                elif k == 'lane_invasion':
                    # 获取车道侵入状态
                    lane_invasion = 1.0 if self.lane_invasion_history[agent_id] > 0 else 0.0
                    observation[k] = np.array([lane_invasion], dtype=np.float32)
                    # 重置车道侵入状态
                    self.lane_invasion_history[agent_id] = 0.0
                else:
                    # 使用原始传感器数据
                    observation[k] = self.sensor_data[agent_id][k]

        # 如果只有一个观察空间，直接返回其内容而非字典
        if len(self.observation_space.spaces) == 1:
            key = next(iter(self.observation_space.spaces))
            return observation[key]

        return observation

    def _compute_reward(self, agent_id):
        """计算指定智能体的奖励"""
        # 获取关键状态信息
        vehicle = self.vehicles[agent_id]

        # 获取速度奖励
        velocity = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # km/h
        speed_reward = self.config.get('reward_speed_coefficient', 0.1) * min(speed, 40.0) / 40.0

        # 获取碰撞惩罚
        collision_penalty = 0.0
        if 'collision' in self.sensor_data[agent_id]:
            collision_intensity = self.sensor_data[agent_id]['collision'][0]
            if collision_intensity > 0.0:
                collision_penalty = self.config.get('reward_collision_penalty', 100.0)

        # 获取车道偏离惩罚
        lane_deviation_penalty = 0.0
        if 'lane_invasion' in self.sensor_data[agent_id] and self.sensor_data[agent_id]['lane_invasion'][0] > 0.0:
            lane_deviation_penalty = self.config.get('reward_lane_deviation_penalty', 0.1)

        # 计算总奖励
        reward = speed_reward - collision_penalty - lane_deviation_penalty

        return reward

    def _check_terminated(self, agent_id):
        """检查指定智能体是否终止"""
        # 检查车辆是否存在
        if agent_id >= len(self.vehicles) or not self.vehicles[agent_id].is_alive:
            return True

        # 检查严重碰撞
        if 'collision' in self.sensor_data[agent_id]:
            collision_intensity = self.sensor_data[agent_id]['collision'][0]
            if collision_intensity > 10.0:  # 临界值可调整
                return True

        return False

    def _check_truncated(self, agent_id):
        """检查指定智能体是否因外部原因截断"""
        # 检查是否达到最大步数
        if self.current_step >= self.max_episode_steps:
            return True

        return False

    def _get_info(self, agent_id):
        """获取指定智能体的附加信息"""
        vehicle = self.vehicles[agent_id]

        # 获取速度
        velocity = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # km/h

        # 获取位置
        location = vehicle.get_location()

        # 获取方向
        rotation = vehicle.get_transform().rotation

        # 获取碰撞计数
        collision_count = 0
        if 'collision' in self.sensor_data[agent_id]:
            collision_intensity = self.sensor_data[agent_id]['collision'][0]
            if collision_intensity > 0.0:
                collision_count = 1

        # 组合信息
        info = {
            'speed': speed,
            'location': (location.x, location.y, location.z),
            'rotation': (rotation.pitch, rotation.yaw, rotation.roll),
            'collision_count': collision_count,
            'lane_invasion_count': self.lane_invasion_history[agent_id],
            'reward': self.last_rewards[agent_id]
        }

        return info

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.pygame_display is not None and 'camera' in self.sensor_data[0]:
                # 获取第一个智能体的相机图像
                array = self.sensor_data[0]['camera']

                # 转换为Pygame表面
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

                # 显示图像
                self.pygame_display.blit(surface, (0, 0))
                pygame.display.flip()

                # 控制帧率
                self.pygame_clock.tick(1.0 / self.delta_seconds)

                # 处理Pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()

        elif self.render_mode == 'rgb_array' and 'camera' in self.sensor_data[0]:
            # 返回RGB数组用于其他可视化
            return self.sensor_data[0]['camera']

    def _process_imu_data(self, agent_id, data):
        """处理IMU数据"""
        # 提取加速度、陀螺仪和指南针数据
        accel = data.accelerometer  # Vector3D
        gyro = data.gyroscope  # Vector3D
        compass = data.compass  # float (in radians)

        # 将数据转换为numpy数组
        imu_data = np.array([
            accel.x, accel.y, accel.z,  # 加速度
            gyro.x, gyro.y, gyro.z,  # 陀螺仪
            compass  # 指南针
        ], dtype=np.float32)

        # 存储数据
        self.sensor_data[agent_id]['imu'] = imu_data

    def close(self):
        """关闭环境并清理资源"""
        # 清除所有车辆和传感器
        self._clear_all_actors()

        # 关闭Pygame
        if self.pygame_display is not None:
            pygame.quit()
            self.pygame_display = None
            self.pygame_clock = None