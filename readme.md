# CARLA Multi-Agent Reinforcement Learning for Autonomous Driving

This project implements a multi-agent reinforcement learning system for autonomous driving using CARLA simulator, Ray, and Stable-Baselines3.

## Environment requirements

- CARLA-0.9.15
- Python 3.7.9
- PyTorch 1.13.0+cu117
- Stable-Baselines3 2.0.0
- Gym 0.23.1
- Ray 2.2.0
- TensorFlow 2.3.1
- TensorBoard 2.1.0

## Project structure

```
carla_marl/
├── agents/ # RL agent implementation
│ ├── __init__.py
│ ├── rl_agent.py # RL agent class
│ └── agent_utils.py # Agent tool function
├── envs/ # Environment wrapper
│ ├── __init__.py
│ ├── carla_env.py # CARLA environment
│ └── env_utils.py # Environment tool function
├── models/ # Model definition
│ ├── __init__.py
│ └── ppo_model.py # PPO model implementation
├── utils/ # Utility tools
│ ├── __init__.py
│ ├── carla_utils.py # CARLA tool function
│ └── visualization.py # Visualization tool
├── configs/ # Configuration file
│ ├── __init__.py
│ ├── env_config.py # Environment configuration
│ ├── agent_config.py # Agent configuration
│ └── train_config.py # Training configuration
├── scripts/ # script
│ ├── __init__.py
│ ├── train.py # training script
│ ├── evaluate.py # evaluation script
│ └── visualize.py # visualization script
└── main.py # main entry program
```

## Features

- Multi-agent reinforcement learning: supports simultaneous learning of multiple vehicles
- Multi-sensor fusion: integrates multiple sensor data such as camera, lidar, radar, GNSS, IMU, etc.
- Flexible environment configuration: customizable maps, weather, sensor parameters, etc.
- Efficient training: use Ray for distributed training
- Visualization tool: supports visualization of training process and evaluation results
- Model evaluation: supports evaluation of model performance on different maps

## Usage

### Environment preparation

Make sure CARLA-0.9.15 is installed and the environment variables are set:

```bash
export CARLA_ROOT=/path/to/carla/ # CARLA root directory
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Training

Run the following command to start training:

```bash
python main.py --mode train --map Town01 --num-agents 4
```

Optional parameters:
- `--map`: Specify the map used for training
- `--num-agents`: Specify the number of agents to be trained
- `--seed`: Random seed
- `--total-timesteps`: Total number of training steps

### Evaluation

Use the trained model for evaluation:

```bash
python main.py --mode evaluate --map Town02 --checkpoint-path ./checkpoints/final_model
```

Optional parameters:
- `--map`: Specify the map used for evaluation
- `--checkpoint-path`: Model checkpoint path

### Visualization

Visualize the behavior of the agent:

```bash
python main.py --mode visualize --map Town03 --checkpoint-path ./checkpoints/final_model
```

## Custom configuration

You can customize system parameters by modifying the following configuration files:

- `configs/env_config.py`: Environment parameter settings
- `configs/agent_config.py`: Agent parameter settings
- `configs/train_config.py`: Training parameter settings

## Main components

### CARLA environment (envs/carla_env.py)

The CARLA environment wrapper implements the Gymnasium interface and provides the following functions:
- Manage the creation and destruction of vehicles and sensors
- Process sensor data
- Calculate rewards and observations
- Detecting termination conditions

### RL Agent (agents/rl_agent.py)

RL Agent class encapsulates the PPO algorithm of Stable-Baselines3:
- Supports multiple sensor inputs
- Custom network architecture
- Provides training, evaluation and prediction interfaces

### Visualization Tool (utils/visualization.py)

The visualization tool provides:
- LiDAR point cloud visualization
- Vehicle information display
- Trajectory drawing
- Reward history drawing

## Training Tips

- You can use a simpler map (such as Town01) at the beginning
- Adjust the reward function appropriately to promote the desired behavior
- Use a longer training time (at least 1 million steps) to get better performance
- Evaluate performance on multiple different maps after training to check generalization ability

## Custom Extensions

### Add new sensors

Modify the `_setup_sensors` method in `envs/carla_env.py` to add new sensor types.

### Modify the reward function

Customize the reward calculation method in the `_compute_reward` method of `envs/carla_env.py`.

### Use different RL algorithms

You can integrate other algorithms of Stable-Baselines3, such as SAC, TD3, etc., by modifying `agents/rl_agent.py`.

## Troubleshooting

- Make sure the CARLA server is running
- Check if the port configuration is correct (default is 2000)
- Check if the CARLA version matches (0.9.15)
- Make sure the graphics driver and CUDA version are compatible

## License

[MIT License](LICENSE)
