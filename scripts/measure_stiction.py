#!/usr/bin/env python3
"""This script measures torques and velocities under motion."""
import os
import time

import numpy as np
import datetime
from ament_index_python.packages import get_package_share_directory

import robot_interfaces
import robot_fingers


def get_random_position():
    """Generate a random position within a save range."""
    initial_position = np.array([0, 0.8, -1.5707])
    position_min = np.array([-0.2, -0.2, -0.2])
    position_max = np.array([0.2, 0.2, 0.2])

    return np.random.uniform(position_min, position_max) + initial_position


def get_torque_lower_joint(t):
    amp_torque = 0.1
    torque = amp_torque * np.sin(-2 * np.pi * t / 2000)
    assert np.abs(torque) < 0.36
    return torque

def get_torque_square_wave(t):
    amp_torque = 0.2
    period = 2000
    phase = (t % period) / period

    if phase < 0.5:
        torque = amp_torque
    else:
        torque = -amp_torque
    assert np.abs(torque) < 0.36
    return torque

def move_finger():
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
    desired_torques = []
    start_time = time.perf_counter()
    count = 0
    max_count = 10001

    while count < max_count:
        desired_torque_lower_joint = get_torque_lower_joint(count)
        #desired_torque_lower_joint = get_torque_square_wave(count)
        desired_torque = np.zeros(3)
        desired_torque[0] = 0.05
        desired_torque[1] = 0.15
        desired_torque[2] = desired_torque_lower_joint

        action = robot_interfaces.finger.Action(torque=desired_torque)
        t = robot_frontend.append_desired_action(action)

        # wait until the action is executed
        robot_frontend.wait_until_timeindex(t)

        # print observation of the current time step
        observation = robot_frontend.get_observation(t)
        timestamps.append(time.perf_counter() - start_time)
        positions.append(observation.position)
        velocities.append(observation.velocity)
        desired_torques.append(desired_torque)
        measured_torques.append(observation.torque)
        count += 1

    positions = np.array(positions)
    velocities = np.array(velocities)
    desired_torques = np.array(desired_torques)
    measured_torques = np.array(measured_torques)
    timestamps = np.array(timestamps)
    data = {
        "positions": positions,
        "velocities": velocities,
        "desired_torques": desired_torques,
        "measured_torques": measured_torques,
        "timestamps": timestamps,
    }
    np.save(f"finger0_ramp_up_torque_motions_lower_joint_{cur_datetime}_no_vel_damping", data)


if __name__ == "__main__":
    move_finger()
