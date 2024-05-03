#!/usr/bin/env python3
"""Basic demo on how to run a Finger Robot with position control."""
import os
import time

import numpy as np
import datetime
from ament_index_python.packages import get_package_share_directory

import robot_interfaces
import robot_fingers


def get_random_position():
    """Generate a random position within a save range."""
    position_min = np.array([-0.2, -0.2, -0.2])
    position_max = np.array([0.2, 0.2, 0.2])

    return np.random.uniform(position_min, position_max)


def demo_position_control():
    # Use the default configuration file from the robot_fingers package
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

    cur_datetime = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    positions = []
    velocities = []
    measured_torques = []
    timestamps = []
    start_time = time.perf_counter()
    count = 0
    max_count = 10000

    while count < max_count:
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
            robot_frontend.wait_until_timeindex(t)

            # print observation of the current time step
            observation = robot_frontend.get_observation(t)
            timestamps.append(time.perf_counter() - start_time)
            positions.append(observation.position)
            velocities.append(observation.velocity)
            measured_torques.append(observation.torque)
            count += 1

    positions = np.array(positions)
    velocities = np.array(velocities)
    measured_torques = np.array(measured_torques)
    timestamps = np.array(timestamps)
    data = {
        "positions": positions,
        "velocities": velocities,
        "measured_torques": measured_torques,
        "timestamps": timestamps,
    }
    np.save(f"random_motions_{cur_datetime}", data)


if __name__ == "__main__":
    demo_position_control()
