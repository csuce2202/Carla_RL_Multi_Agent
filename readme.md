# CARLA Multi-Agent Reinforcement Learning for Autonomous Driving

This project implements a multi-agent reinforcement learning system for autonomous driving using CARLA simulator, Ray, and Stable-Baselines3.

## 环境要求

- CARLA-0.9.15
- Python 3.7.9
- PyTorch 1.13.0+cu117
- Stable-Baselines3 2.0.0
- Gym 0.23.1
- Ray 2.2.0
- TensorFlow 2.3.1
- TensorBoard 2.1.0

## 项目结构

```
carla_marl/
├── agents/                 # RL智能体实现
│   ├── __init__.py
│   ├── rl_agent.py         # RL智能体类
│   └── agent_utils.py      # 智能体工具函数
├── envs/                   # 环境包装器
│   ├── __init__.py
│   ├── carla_env.py        # CARLA环境
│   └── env_utils.py        # 环境工具函数
├── models/                 # 模型定义
│   ├── __init__.py
│   └── ppo_model.py        # PPO模型实现
├── utils/                  # 实用工具
│   ├── __init__.py
│   ├── carla_utils.py      # CARLA工具函数
│   └── visualization.py    # 可视化工具
├── configs/                # 配置文件
│   ├── __init__.py
│   ├── env_config.py       # 环境配置
│   ├── agent_config.py     # 智能体配置
│   └── train_config.py     # 训练配置
├── scripts/                # 脚本
│   ├── __init__.py
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   └── visualize.py        # 可视化脚本
└── main.py                 # 主入口程序
```

## 功能特性

- 多智能体强化学习：支持多车辆同时学习
- 多传感器融合：整合相机、激光雷达、雷达、GNSS、IMU等多种传感器数据
- 灵活的环境配置：可自定义地图、天气、传感器参数等
- 高效的训练：使用Ray进行分布式训练
- 可视化工具：支持训练过程和评估结果的可视化
- 模型评估：支持在不同地图上评估模型性能

## 使用方法

### 环境准备

确保已安装CARLA-0.9.15，并设置环境变量：

```bash
export CARLA_ROOT=/path/to/carla/  # CARLA根目录
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

安装依赖:

```bash
pip install -r requirements.txt
```

### 训练

运行以下命令开始训练：

```bash
python main.py --mode train --map Town01 --num-agents 4
```

可选参数:
- `--map`：指定训练使用的地图
- `--num-agents`：指定训练的智能体数量
- `--seed`：随机种子
- `--total-timesteps`：训练总步数

### 评估

使用训练好的模型进行评估：

```bash
python main.py --mode evaluate --map Town02 --checkpoint-path ./checkpoints/final_model
```

可选参数:
- `--map`：指定评估使用的地图
- `--checkpoint-path`：模型检查点路径

### 可视化

可视化智能体的行为：

```bash
python main.py --mode visualize --map Town03 --checkpoint-path ./checkpoints/final_model
```

## 自定义配置

可以通过修改以下配置文件来自定义系统参数：

- `configs/env_config.py`：环境参数设置
- `configs/agent_config.py`：智能体参数设置
- `configs/train_config.py`：训练参数设置

## 主要组件

### CARLA环境 (envs/carla_env.py)

CARLA环境封装器实现了Gymnasium接口，提供了以下功能：
- 管理车辆和传感器的创建和销毁
- 处理传感器数据
- 计算奖励和观察
- 检测终止条件

### RL智能体 (agents/rl_agent.py)

RL智能体类封装了Stable-Baselines3的PPO算法：
- 支持多传感器输入
- 自定义网络架构
- 提供了训练、评估和预测接口

### 可视化工具 (utils/visualization.py)

可视化工具提供了：
- 激光雷达点云可视化
- 车辆信息显示
- 轨迹绘制
- 奖励历史绘制

## 训练提示

- 开始时可以使用较简单的地图（如Town01）
- 适当调整奖励函数以促进期望行为
- 使用较长的训练时间（至少100万步）获得较好的性能
- 训练后在多个不同地图上评估性能，检查泛化能力

## 自定义扩展

### 添加新的传感器

修改`envs/carla_env.py`中的`_setup_sensors`方法添加新的传感器类型。

### 修改奖励函数

在`envs/carla_env.py`的`_compute_reward`方法中自定义奖励计算方式。

### 使用不同的RL算法

可以通过修改`agents/rl_agent.py`，集成Stable-Baselines3的其他算法，如SAC、TD3等。

## 故障排除

- 确保CARLA服务器正在运行
- 检查端口配置是否正确（默认为2000）
- 检查CARLA版本是否匹配（0.9.15）
- 确保显卡驱动和CUDA版本兼容

## 许可证

[MIT License](LICENSE)
