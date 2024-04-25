#!/usr/bin/env python3
"""Basic demo on how to run a Finger Robot with position control."""
import os
import numpy as np
import time
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor

import rclpy
import robot_interfaces
import robot_fingers
from robot_fingers.ros import TrifingerActionSubscriberStatePublisher


def get_random_position():
    """Generate a random position within a save range."""
    position_min = np.array([-0.1, -0.0, -0.0])
    position_max = np.array([0.1, 0.0, 0.0])

    return np.random.uniform(position_min, position_max)


def demo_position_control():
    # Use the default configuration file from the robot_fingers package
    rclpy.init()

    config_file_path = os.path.join(
        get_package_share_directory("robot_fingers"), "config", "finger.yml"
    )

    # Storage for all observations, actions, etc.
    robot_data = robot_interfaces.finger.SingleProcessData()

    # The backend takes care of communication with the robot hardware.
    robot_backend = robot_fingers.create_real_finger_backend(
        robot_data, config_file_path
    )

    # The frontend is used by the user to get observations and send actions
    robot_frontend = robot_interfaces.finger.Frontend(robot_data)

    trifinger_act_sub_state_pub = TrifingerActionSubscriberStatePublisher(
        robot_frontend
    )
    executor = MultiThreadedExecutor()
    executor.add_node(trifinger_act_sub_state_pub)

    # Initializes the robot (e.g. performs homing).
    robot_backend.initialize()

    # Start spinning ROS processes until shutdown.
    executor.spin()
    executor.shutdown()
    trifinger_act_sub_state_pub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    os.nice(-19)
    demo_position_control()
