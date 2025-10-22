import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from ros_gz_interfaces.msg import EntityWrench , Entity
import subprocess
import time
from typing import Optional


class CartPoleGazebo(gym.Env):
    def __init__(self):
        super().__init__()

        #find directory and launch
        subprocess.run("cd /home/flow/reinforcement_learning/gazebo_sim/cartpole_balancer && ./spawn_single_model.sh", shell=True)

        # Initialize ROS client library and node
        rclpy.init(args=None)
        self.node = Node("cartpole_gazebo_env")

        # ROS subscriptions and publishers
        self.obs_sub = self.node.create_subscription(
            JointState,
            'joint_states',   # You had 'sensor_msgs/msg/JointState' which is the msg type, but topic name is needed here
            self.joint_state_callback,
            10
        )
        self.force_pub = self.node.create_publisher(EntityWrench, '/apply_force_and_torque', 10)

        # Internal state
        self.joint_position = 0.0
        self.joint_velocity = 0.0
        self.received_state = False

        # Action and observation spaces
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(2,), dtype=np.float32)

        # Reward tracking
        self.reward = 0.0
        self.max_steps = 500
        self.current_step = 0

    def joint_state_callback(self, msg: JointState):
        # Update internal state from sensor data
        self.joint_position = msg.position[0]
        self.joint_velocity = msg.velocity[0]
        self.received_state = True

    def _get_observation(self):
        return np.array([self.joint_position, self.joint_velocity], dtype=np.float32)

    def _get_info(self):
        position_error = np.abs(self.joint_position)
        return {"position_error": position_error}

    def reset(self, seed:Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Reset simulation through gz service call
        cmd = [
            "gz", "service", "-s", "/world/empty/control",
            "--reqtype", "gz.msgs.WorldControl",
            "--reptype", "gz.msgs.Boolean",
            "--req", "reset { all: true }"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Simulation reset successfully!")
            print("Response:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Failed to reset simulation.")
            print("Error:", e.stderr)

        # Wait until we receive the new state after reset
        self.received_state = False
        timeout = 5.0  # seconds
        start_time = time.time()
        while not self.received_state and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        self.reward = 0.0
        self.current_step = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.current_step += 1

        # Publish applied force as EntityWrench message
        robot = Entity()
        robot.name = "base_link"
        robot.id = 9
        wrench_msg = EntityWrench()
        wrench_msg.entity = robot
        wrench_msg.wrench.force.x = float(action[0])
        wrench_msg.wrench.force.y = 0.0
        wrench_msg.wrench.force.z = 0.0
        wrench_msg.wrench.torque.x = 0.0
        wrench_msg.wrench.torque.y = 0.0
        wrench_msg.wrench.torque.z = 0.0

        self.force_pub.publish(wrench_msg)

        # Spin node to process incoming messages
        rclpy.spin_once(self.node)

        observation = self._get_observation()
        info = self._get_info()

        # Reward: encourage keeping joint near zero position
        position_error = info["position_error"]
        reward = -position_error  # Negative of error, the closer to zero, the better
        self.reward += reward

        # Termination conditions
        terminated = False
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        if abs(self.joint_position) > 10:  # example failure condition if joint position too large
            terminated = True

        return observation, reward, terminated, truncated, info

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

