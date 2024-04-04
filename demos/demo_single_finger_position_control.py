#!/usr/bin/env python3
"""Basic demo on how to run a Finger Robot with position control."""
import os
import numpy as np
from ament_index_python.packages import get_package_share_directory

import rclpy
import robot_interfaces
import robot_fingers
from robot_fingers.ros import TrifingerStatePublisher


def get_random_position():
    """Generate a random position within a save range."""
    position_min = np.array([-0.0, -0.0, -0.0])
    position_max = np.array([0.0, 0.0, 0.0])

    return np.random.uniform(position_min, position_max)


def demo_position_control():
    # Use the default configuration file from the robot_fingers package
    rclpy.init()
    trifinger_state_publisher = TrifingerStatePublisher()
    
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

    # Initializes the robot (e.g. performs homing).
    robot_backend.initialize()

    while True:
        # Run a position controller that randomly changes the desired position
        # every 500 steps.  One time step corresponds to roughly 1 ms.

        desired_position = get_random_position()
        for _ in range(500):
            # Appends a torque command ("action") to the action queue.
            # Returns the time step at which the action is going to be
            # executed.
            action = robot_interfaces.finger.Action(position=desired_position)
            t = robot_frontend.append_desired_action(action)

            # wait until the action is executed
            #robot_frontend.wait_until_timeindex(t)
            obs = robot_frontend.get_observation(t)
            trifinger_state_publisher.publish(obs)
            rclpy.spin_once(trifinger_state_publisher, timeout_sec=1e-4)

        # print observation of the current time step
        #observation = robot_frontend.get_observation(t)
        #print("-----")
        #print("Position: %s" % observation.position)
        #print("Velocity: %s" % observation.velocity)
        #print("Torque: %s" % observation.torque)

    trifinger_state_publisher.destroy_node()
    rclpy.shut_down()


if __name__ == "__main__":
    os.nice(-19)
    demo_position_control()
