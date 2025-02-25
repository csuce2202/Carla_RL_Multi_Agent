import os
import time
import random
import math
import numpy as np
import carla
from pathlib import Path


def wait_for_carla(host='localhost', port=2000, timeout=20.0):
    """
    等待CARLA服务器启动

    参数:
        host: 服务器主机名
        port: 服务器端口
        timeout: 超时时间(秒)

    返回:
        carla.Client: 已连接的客户端
    """
    import socket

    # 尝试连接
    start_time = time.time()
    while True:
        try:
            # 尝试创建客户端连接
            client = carla.Client(host, port)
            client.set_timeout(5.0)  # 短连接超时

            # 如果能获取世界，则连接成功
            world = client.get_world()
            print(f"成功连接到CARLA服务器: {world.get_map().name}")

            # 设置正常超时
            client.set_timeout(timeout)
            return client

        except (socket.error, RuntimeError) as e:
            # 检查是否超时
            if time.time() - start_time > timeout:
                raise TimeoutError(f"连接CARLA服务器超时，请确保服务器已启动: {e}")

            print(f"等待CARLA服务器启动... ({int(time.time() - start_time)}s)")
            time.sleep(2.0)


def start_carla_server(port=2000, low_quality=False, no_rendering=False):
    """
    启动CARLA服务器

    参数:
        port: 服务器端口
        low_quality: 是否使用低质量渲染
        no_rendering: 是否禁用渲染

    返回:
        subprocess.Popen: 服务器进程
    """
    import subprocess
    import platform

    # 获取CARLA路径
    carla_path = os.environ.get('CARLA_ROOT')
    if carla_path is None:
        raise ValueError("环境变量CARLA_ROOT未设置，请设置CARLA根目录路径")

    # 确定可执行文件路径
    if platform.system() == 'Windows':
        carla_exec = os.path.join(carla_path, 'CarlaUE4.exe')
    else:
        carla_exec = os.path.join(carla_path, 'CarlaUE4.sh')

    # 检查可执行文件是否存在
    if not os.path.exists(carla_exec):
        raise FileNotFoundError(f"CARLA可执行文件不存在: {carla_exec}")

    # 构建命令
    cmd = [carla_exec, '-carla-rpc-port=' + str(port)]

    if low_quality:
        cmd.append('-quality-level=Low')

    if no_rendering:
        cmd.append('-RenderOffScreen')

    # 启动服务器
    print(f"启动CARLA服务器: {' '.join(cmd)}")
    server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 等待服务器启动
    time.sleep(5.0)

    return server_process


def stop_carla_server(server_process):
    """
    停止CARLA服务器

    参数:
        server_process: 服务器进程
    """
    if server_process is not None:
        server_process.terminate()
        server_process.wait()
        print("CARLA服务器已停止")


def find_spawn_points(world, n_points=None, min_distance=10.0):
    """
    查找不重叠的生成点

    参数:
        world: CARLA世界
        n_points: 要返回的生成点数量，None表示尽可能多
        min_distance: 生成点之间的最小距离

    返回:
        list: 生成点列表
    """
    # 获取所有生成点
    spawn_points = world.get_map().get_spawn_points()

    if n_points is None or n_points >= len(spawn_points):
        return spawn_points

    # 随机洗牌
    random.shuffle(spawn_points)

    # 选择不重叠的生成点
    selected_points = [spawn_points[0]]

    for point in spawn_points[1:]:
        # 检查与已选择的点的距离
        too_close = False
        for selected in selected_points:
            dist = point.location.distance(selected.location)
            if dist < min_distance:
                too_close = True
                break

        # 如果不太近，则添加
        if not too_close:
            selected_points.append(point)

            # 如果已经有足够的点，则返回
            if len(selected_points) >= n_points:
                break

    return selected_points


def create_traffic(client, world, n_vehicles=50, n_pedestrians=20, safe_spawn=True):
    """
    创建交通(车辆和行人)

    参数:
        client: CARLA客户端
        world: CARLA世界
        n_vehicles: 要生成的车辆数量
        n_pedestrians: 要生成的行人数量
        safe_spawn: 是否使用安全生成

    返回:
        tuple: (车辆列表, 行人列表, 行人控制器列表)
    """
    # 获取车辆生成点
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    # 设置车辆生成参数
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]
    blueprints = [bp for bp in blueprints if not bp.id.endswith('ambulance') and not bp.id.endswith('firetruck')]

    # 生成车辆
    batch = []
    vehicles = []
    for i, transform in enumerate(spawn_points):
        if i >= n_vehicles:
            break

        blueprint = random.choice(blueprints)

        # 设置车辆颜色
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # 设置是否碰撞检测
        blueprint.set_attribute('role_name', 'autopilot')

        # 在安全模式下，我们需要确保不会生成在其他车辆的位置
        if safe_spawn:
            transform.location.z += 0.5  # 避免车辆嵌入地面

            if world.try_spawn_actor(blueprint, transform):
                batch.append(carla.command.SpawnActor(blueprint, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True)))
        else:
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)))

    # 批量生成车辆
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(f"车辆生成错误: {response.error}")
        else:
            vehicles.append(response.actor_id)

    vehicles = world.get_actors(vehicles)
    print(f"生成了 {len(vehicles)} 辆车")

    # 设置行人生成参数
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

    # 生成行人
    batch = []
    pedestrians = []
    pedestrian_controllers = []

    for i in range(n_pedestrians):
        blueprint = random.choice(blueprintsWalkers)

        # 设置行人属性
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')

        # 生成行人
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()

        if spawn_point.location is not None:
            batch.append(carla.command.SpawnActor(blueprint, spawn_point))

    # 批量生成行人
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(f"行人生成错误: {response.error}")
        else:
            pedestrians.append(response.actor_id)

    pedestrians = world.get_actors(pedestrians)
    print(f"生成了 {len(pedestrians)} 个行人")

    # 生成行人控制器
    batch = []
    controller_bp = world.get_blueprint_library().find('controller.ai.walker')

    for pedestrian in pedestrians:
        batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), pedestrian))

    # 批量生成控制器
    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(f"行人控制器生成错误: {response.error}")
        else:
            pedestrian_controllers.append(response.actor_id)

    pedestrian_controllers = world.get_actors(pedestrian_controllers)

    # 启动行人
    for controller in pedestrian_controllers:
        # 设置随机目标点
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        # 设置随机行走速度
        controller.set_max_speed(1 + random.random())

    return vehicles, pedestrians, pedestrian_controllers


def destroy_traffic(client, vehicles, pedestrians, pedestrian_controllers):
    """
    销毁交通

    参数:
        client: CARLA客户端
        vehicles: 车辆列表
        pedestrians: 行人列表
        pedestrian_controllers: 行人控制器列表
    """
    # 停止所有行人
    for controller in pedestrian_controllers:
        controller.stop()

    # 批量销毁车辆和行人
    batch = []

    for vehicle in vehicles:
        batch.append(carla.command.DestroyActor(vehicle))

    for controller in pedestrian_controllers:
        batch.append(carla.command.DestroyActor(controller))

    for pedestrian in pedestrians:
        batch.append(carla.command.DestroyActor(pedestrian))

    client.apply_batch_sync(batch, True)
    print(f"已销毁 {len(vehicles)} 辆车和 {len(pedestrians)} 个行人")


def get_vehicle_bbox(vehicle):
    """
    获取车辆的边界框

    参数:
        vehicle: 车辆Actor

    返回:
        tuple: (x_min, y_min, z_min, x_max, y_max, z_max)
    """
    # 获取车辆变换和边界框
    transform = vehicle.get_transform()
    bbox = vehicle.bounding_box

    # 计算边界框的8个顶点
    vertices = []
    for x in [-bbox.extent.x, bbox.extent.x]:
        for y in [-bbox.extent.y, bbox.extent.y]:
            for z in [-bbox.extent.z, bbox.extent.z]:
                vertices.append(carla.Location(x=x, y=y, z=z))

                # 将顶点从局部坐标转换为世界坐标
                world_vertices = []
                for vertex in vertices:
                    world_vertex = transform.transform(vertex)
                    world_vertices.append(world_vertex)

                # 找到边界框的最小和最大坐标
                x_min = min(vertex.x for vertex in world_vertices)
                y_min = min(vertex.y for vertex in world_vertices)
                z_min = min(vertex.z for vertex in world_vertices)
                x_max = max(vertex.x for vertex in world_vertices)
                y_max = max(vertex.y for vertex in world_vertices)
                z_max = max(vertex.z for vertex in world_vertices)

                return (x_min, y_min, z_min, x_max, y_max, z_max)

            def get_2d_bbox(vehicle):
                """
                获取车辆的2D边界框

                参数:
                    vehicle: 车辆Actor

                返回:
                    tuple: (x_min, y_min, x_max, y_max)
                """
                x_min, y_min, _, x_max, y_max, _ = get_vehicle_bbox(vehicle)
                return (x_min, y_min, x_max, y_max)

            def get_relative_transform(transform, reference_transform):
                """
                计算相对于参考变换的相对变换

                参数:
                    transform: 当前变换
                    reference_transform: 参考变换

                返回:
                    carla.Transform: 相对变换
                """
                # 计算相对位置
                forward_vector = reference_transform.get_forward_vector()
                right_vector = reference_transform.get_right_vector()
                up_vector = reference_transform.get_up_vector()

                pos_diff = transform.location - reference_transform.location

                # 计算相对坐标
                x = (pos_diff.x * forward_vector.x +
                     pos_diff.y * forward_vector.y +
                     pos_diff.z * forward_vector.z)

                y = (pos_diff.x * right_vector.x +
                     pos_diff.y * right_vector.y +
                     pos_diff.z * right_vector.z)

                z = (pos_diff.x * up_vector.x +
                     pos_diff.y * up_vector.y +
                     pos_diff.z * up_vector.z)

                # 计算相对旋转
                rel_yaw = transform.rotation.yaw - reference_transform.rotation.yaw
                rel_pitch = transform.rotation.pitch - reference_transform.rotation.pitch
                rel_roll = transform.rotation.roll - reference_transform.rotation.roll

                # 创建相对变换
                relative_location = carla.Location(x=x, y=y, z=z)
                relative_rotation = carla.Rotation(pitch=rel_pitch, yaw=rel_yaw, roll=rel_roll)

                return carla.Transform(relative_location, relative_rotation)

            def distance_to_line(point, line_start, line_end):
                """
                计算点到线段的距离

                参数:
                    point: 点(carla.Location)
                    line_start: 线段起点(carla.Location)
                    line_end: 线段终点(carla.Location)

                返回:
                    float: 距离
                """
                # 线段长度
                length_squared = (line_end.x - line_start.x) ** 2 + (line_end.y - line_start.y) ** 2

                # 如果线段长度为0，则返回点到起点的距离
                if length_squared == 0:
                    return math.sqrt((point.x - line_start.x) ** 2 + (point.y - line_start.y) ** 2)

                # 计算投影点参数
                t = max(0, min(1, ((point.x - line_start.x) * (line_end.x - line_start.x) +
                                   (point.y - line_start.y) * (line_end.y - line_start.y)) / length_squared))

                # 计算投影点
                projection_x = line_start.x + t * (line_end.x - line_start.x)
                projection_y = line_start.y + t * (line_end.y - line_start.y)

                # 计算距离
                distance = math.sqrt((point.x - projection_x) ** 2 + (point.y - projection_y) ** 2)

                return distance

            def get_waypoint_path(waypoint, distance=5.0, length=20):
                """
                获取从给定路点开始的路径

                参数:
                    waypoint: 起始路点
                    distance: 相邻路点之间的距离
                    length: 要返回的路点数量

                返回:
                    list: 路点列表
                """
                path = [waypoint]

                for _ in range(length - 1):
                    next_waypoints = path[-1].next(distance)
                    if not next_waypoints:
                        break
                    path.append(next_waypoints[0])

                return path

            def get_actor_display_name(actor):
                """
                获取Actor的显示名称

                参数:
                    actor: CARLA Actor

                返回:
                    str: 显示名称
                """
                name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
                return name

            def get_speed(vehicle):
                """
                获取车辆的速度(km/h)

                参数:
                    vehicle: 车辆Actor

                返回:
                    float: 速度(km/h)
                """
                velocity = vehicle.get_velocity()
                # 计算速度(km/h)
                return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

            def get_acceleration(vehicle):
                """
                获取车辆的加速度(m/s^2)

                参数:
                    vehicle: 车辆Actor

                返回:
                    float: 加速度(m/s^2)
                """
                acceleration = vehicle.get_acceleration()
                return math.sqrt(acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2)

            def get_traffic_light_state(vehicle, world):
                """
                获取车辆前方红绿灯的状态

                参数:
                    vehicle: 车辆Actor
                    world: CARLA世界

                返回:
                    carla.TrafficLightState: 红绿灯状态，如果没有红绿灯则为None
                """
                # 获取车辆位置和朝向
                transform = vehicle.get_transform()
                forward_vector = transform.get_forward_vector()

                # 投射射线，检测前方红绿灯
                ray_length = 50.0
                end_point = transform.location + forward_vector * ray_length

                # 获取交通灯
                traffic_lights = world.get_actors().filter('traffic.traffic_light*')

                # 最近的红绿灯及其距离
                nearest_light = None
                min_distance = float('inf')

                for traffic_light in traffic_lights:
                    # 计算红绿灯到车辆的距离
                    light_location = traffic_light.get_location()
                    distance = transform.location.distance(light_location)

                    # 检查红绿灯是否在车辆前方
                    vector_to_light = light_location - transform.location
                    angle = math.degrees(math.acos(
                        (forward_vector.x * vector_to_light.x + forward_vector.y * vector_to_light.y) /
                        (math.sqrt(forward_vector.x ** 2 + forward_vector.y ** 2) *
                         math.sqrt(vector_to_light.x ** 2 + vector_to_light.y ** 2))
                    ))

                    # 如果红绿灯在车辆前方30度范围内，更新最近的红绿灯
                    if angle < 30.0 and distance < min_distance:
                        nearest_light = traffic_light
                        min_distance = distance

                if nearest_light is not None:
                    return nearest_light.state

                return None

            def get_road_info(vehicle, map):
                """
                获取车辆所在道路的信息

                参数:
                    vehicle: 车辆Actor
                    map: CARLA地图

                返回:
                    dict: 道路信息
                """
                # 获取车辆位置
                location = vehicle.get_location()

                # 获取最近的路点
                waypoint = map.get_waypoint(location)

                # 收集道路信息
                road_info = {
                    'road_id': waypoint.road_id,
                    'lane_id': waypoint.lane_id,
                    'lane_type': waypoint.lane_type,
                    'is_junction': waypoint.is_junction,
                    'lane_width': waypoint.lane_width,
                    'lane_change': waypoint.lane_change,
                }

                return road_info

            def is_vehicle_on_lane(vehicle, map, max_distance=1.0):
                """
                检查车辆是否在车道上

                参数:
                    vehicle: 车辆Actor
                    map: CARLA地图
                    max_distance: 最大允许偏离距离

                返回:
                    bool: 是否在车道上
                """
                # 获取车辆位置
                location = vehicle.get_location()

                # 获取最近的路点
                waypoint = map.get_waypoint(location)

                # 计算车辆到路点的距离
                distance = location.distance(waypoint.transform.location)

                return distance <= max_distance

            def draw_waypoints(world, waypoints, lifetime=5.0):
                """
                在世界中绘制路点

                参数:
                    world: CARLA世界
                    waypoints: 路点列表
                    lifetime: 显示时间(秒)
                """
                for waypoint in waypoints:
                    world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                            color=carla.Color(r=0, g=255, b=0), life_time=lifetime)

            def save_images(image_list, output_dir, prefix='image', extension='png'):
                """
                保存图像列表

                参数:
                    image_list: 图像列表(numpy数组)
                    output_dir: 输出目录
                    prefix: 文件名前缀
                    extension: 文件扩展名
                """
                import cv2

                # 创建输出目录
                os.makedirs(output_dir, exist_ok=True)

                for i, image in enumerate(image_list):
                    filename = f"{prefix}_{i:04d}.{extension}"
                    filepath = os.path.join(output_dir, filename)

                    # 将BGR转换为RGB(如果需要)
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(filepath, image)

                print(f"已保存 {len(image_list)} 张图像到 {output_dir}")

            def set_weather(world, preset_name):
                """
                设置天气预设

                参数:
                    world: CARLA世界
                    preset_name: 预设名称

                返回:
                    bool: 是否成功设置
                """
                # 天气预设
                presets = {
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

                if preset_name in presets:
                    world.set_weather(presets[preset_name])
                    return True

                print(f"未知的天气预设: {preset_name}")
                return False

            def create_custom_weather(world, cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,
                                      wind_intensity=0.0, sun_azimuth_angle=0.0, sun_altitude_angle=70.0,
                                      fog_density=0.0, fog_distance=0.0, wetness=0.0):
                """
                创建自定义天气

                参数:
                    world: CARLA世界
                    以及各种天气参数
                """
                weather = carla.WeatherParameters(
                    cloudiness=cloudiness,
                    precipitation=precipitation,
                    precipitation_deposits=precipitation_deposits,
                    wind_intensity=wind_intensity,
                    sun_azimuth_angle=sun_azimuth_angle,
                    sun_altitude_angle=sun_altitude_angle,
                    fog_density=fog_density,
                    fog_distance=fog_distance,
                    wetness=wetness
                )

                world.set_weather(weather)
                return weather