AGENT_CONFIG = {
    "num_agents": 4,
    "agent_type": "ppo",
    "discount_factor": 0.99,
    "learning_rate": 3e-4,
    "observation_space": {
        "camera": True,
        "lidar": True,
        "radar": False,
        "gnss": True,
        "imu": True,
        "speedometer": True,
        "lane_invasion": True,
        "collision": True,
    },
    "action_space": "continuous",  # 或者 "discrete"
    "action_smoothing": 0.9,
    "stacked_frames": 4
}