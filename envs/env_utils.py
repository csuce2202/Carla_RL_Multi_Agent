import numpy as np
import random
import carla
import math
from pathlib import Path


def get_vehicle_waypoint(world, vehicle):
    """获取车辆当前所在的路点"""
    location = vehicle.get_location()
    return world.get_map().get_waypoint(location)


def is_within_distance(target_location, reference_location, max_distance):
    """检查两个位置是否在指定距离内"""
    dx = target_location.x - reference_location.x
    dy = target_location.y - reference_location.y
    dz = target_location.z - reference_location.z

    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    return distance <= max_distance


def compute_distance(location_1, location_2):
    """计算两个位置之间的距离"""
    dx = location_1.x - location_2.x
    dy = location_1.y - location_2.y
    dz = location_1.z - location_2.z

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def preprocess_image(image, height=84, width=84):
    """预处理图像用于模型输入"""
    import cv2

    # 调整大小
    image = cv2.resize(image, (width, height))

    # 转换为灰度图（如果需要）
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 归一化到0-1范围
    image = image / 255.0

    return image


def filter_lidar_data(lidar_data, max_distance=50.0, min_height=-2.0, max_height=5.0):
    """过滤激光雷达数据，去除距离过远和高度不合适的点"""
    # 过滤距离
    distance_mask = np.sqrt(np.sum(lidar_data[:, :2] ** 2, axis=1)) <= max_distance

    # 过滤高度
    height_mask = (lidar_data[:, 2] >= min_height) & (lidar_data[:, 2] <= max_height)

    # 组合掩码
    mask = distance_mask & height_mask

    return lidar_data[mask]


def voxelize_lidar(lidar_data, voxel_size=(10, 10, 8), point_cloud_range=(-50, -50, -5, 50, 50, 3)):
    """将激光雷达点云体素化"""
    # 点云范围
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    # 体素大小
    x_voxels, y_voxels, z_voxels = voxel_size

    # 创建空体素网格
    voxel_grid = np.zeros(voxel_size, dtype=np.float32)

    # 计算每个维度的步长
    x_step = (x_max - x_min) / x_voxels
    y_step = (y_max - y_min) / y_voxels
    z_step = (z_max - z_min) / z_voxels

    # 计算每个点所属的体素索引
    x_indices = np.floor((lidar_data[:, 0] - x_min) / x_step).astype(np.int32)
    y_indices = np.floor((lidar_data[:, 1] - y_min) / y_step).astype(np.int32)
    z_indices = np.floor((lidar_data[:, 2] - z_min) / z_step).astype(np.int32)

    # 过滤超出范围的点
    valid_mask = (
            (x_indices >= 0) & (x_indices < x_voxels) &
            (y_indices >= 0) & (y_indices < y_voxels) &
            (z_indices >= 0) & (z_indices < z_voxels)
    )

    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    z_indices = z_indices[valid_mask]

    # 填充体素网格
    for i in range(len(x_indices)):
        voxel_grid[x_indices[i], y_indices[i], z_indices[i]] += 1.0

    return voxel_grid


def get_random_blueprint(world, vehicle_type='car'):
    """获取随机的车辆蓝图"""
    if vehicle_type == 'car':
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        # 过滤出四轮车
        blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]
    elif vehicle_type == 'bike':
        blueprints = world.get_blueprint_library().filter('vehicle.bh.crossbike')
    elif vehicle_type == 'pedestrian':
        blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    else:
        blueprints = world.get_blueprint_library().filter('vehicle.*')

    return random.choice(blueprints)


def find_weather_presets():
    """查找CARLA内置的天气预设"""
    presets = [x for x in dir(carla.WeatherParameters) if
               isinstance(getattr(carla.WeatherParameters, x), carla.WeatherParameters)]
    return presets


def create_traffic(world, num_vehicles=50, num_pedestrians=20):
    """创建交通（车辆和行人）"""
    # 获取所有可用的生成点
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # 确保有足够的生成点
    if len(spawn_points) < num_vehicles:
        num_vehicles = len(spawn_points)

    # 创建车辆
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    vehicle_blueprints = [bp for bp in vehicle_blueprints if int(bp.get_attribute('number_of_wheels')) == 4]

    vehicles = []
    for i in range(num_vehicles):
        blueprint = random.choice(vehicle_blueprints)

        # 设置颜色
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # 尝试生成车辆
        try:
            vehicle = world.spawn_actor(blueprint, spawn_points[i])
            vehicles.append(vehicle)
        except:
            pass

    # 设置车辆行为
    for vehicle in vehicles:
        vehicle.set_autopilot(True)

    # 创建行人
    pedestrian_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')

    pedestrians = []
    pedestrian_controllers = []

    for i in range(num_pedestrians):
        blueprint = random.choice(pedestrian_blueprints)

        # 设置是否跑步
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')

        # 在随机位置生成行人
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()

        if spawn_point.location:
            try:
                pedestrian = world.spawn_actor(blueprint, spawn_point)
                pedestrians.append(pedestrian)
            except:
                pass

    # 为行人创建AI控制器
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')

    for pedestrian in pedestrians:
        controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
        controller.start()
        pedestrian_controllers.append(controller)

    return vehicles, pedestrians, pedestrian_controllers


def clean_traffic(world, vehicles, pedestrians, pedestrian_controllers):
    """清理交通（车辆和行人）"""
    # 停止行人AI控制器
    for controller in pedestrian_controllers:
        controller.stop()

    # 清理车辆和行人
    for actor in vehicles + pedestrians + pedestrian_controllers:
        if actor.is_alive:
            actor.destroy()


def save_image(image, folder, filename):
    """保存图像到文件"""
    import cv2
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / filename), image[:, :, ::-1])  # RGB to BGR for OpenCV


def generate_simple_waypoints(world, num_waypoints=10, road_id=None, lane_id=None):
    """生成简单的路点序列用于导航"""
    if road_id is None or lane_id is None:
        # 随机选择一条道路和车道
        waypoints = world.get_map().generate_waypoints(distance=2.0)
        if not waypoints:
            return []

        start_waypoint = random.choice(waypoints)
        road_id = start_waypoint.road_id
        lane_id = start_waypoint.lane_id
    else:
        # 使用指定的道路和车道
        waypoints = world.get_map().generate_waypoints(distance=2.0)
        filtered_waypoints = [wp for wp in waypoints if wp.road_id == road_id and wp.lane_id == lane_id]

        if not filtered_waypoints:
            return []

        start_waypoint = filtered_waypoints[0]

    # 生成连续的路点
    next_waypoints = [start_waypoint]
    for _ in range(num_waypoints - 1):
        next_wps = next_waypoints[-1].next(2.0)
        if not next_wps:
            break
        next_waypoints.append(next_wps[0])

    return next_waypoints