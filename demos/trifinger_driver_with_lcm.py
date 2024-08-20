#!/usr/bin/env python3
"""Basic demo on how to run a Finger Robot with torque control."""
import os
import select
import lcm
import numpy as np

import robot_interfaces
import robot_fingers

from ament_index_python.packages import get_package_share_directory
from lcmt_robot_input import lcmt_robot_input
from lcmt_robot_output import lcmt_robot_output


JOINT_POSITION_NAMES = [
    "finger_base_to_upper_joint_0",
    "finger_upper_to_middle_joint_0",
    "finger_middle_to_lower_joint_0",
    "finger_base_to_upper_joint_120",
    "finger_upper_to_middle_joint_120",
    "finger_middle_to_lower_joint_120",
    "finger_base_to_upper_joint_240",
    "finger_upper_to_middle_joint_240",
    "finger_middle_to_lower_joint_240",
]

JOINT_VELOCITY_NAMES = [joint_name + "dot" for joint_name in JOINT_POSITION_NAMES]
JOINT_TORQUE_NAMES = [joint_name + "_torque" for joint_name in JOINT_POSITION_NAMES]


class TrifingerInputSubscriber:
    def __init__(self, lcm_obj, channel):
        self.lcm_obj = lcm_obj
        self.channel = channel
        self.trifinger_inputs = np.zeros(9)
        self.lcm_obj.subscribe(self.channel, self.callback)

    def callback(self, channel, data):
        msg = lcmt_robot_input.decode(data)
        self.trifinger_inputs = np.array(msg.efforts)

    def get_input(self):
        return self.trifinger_inputs


class TrifingerStatePublisher:
    def __init__(self, lcm_obj, channel):
        self.lcm_obj = lcm_obj
        self.channel = channel

    def pub_observation(self, observation):
        msg = lcmt_robot_output()
        msg.num_positions = 9
        msg.num_velocities = 9
        msg.num_efforts = 9
        msg.position_names = JOINT_POSITION_NAMES
        msg.velocity_names = JOINT_VELOCITY_NAMES
        msg.effort_names = JOINT_TORQUE_NAMES
        msg.position = np.array(observation.position)
        msg.velocity = np.array(observation.velocity)
        msg.effort = np.array(observation.torque)
        self.lcm_obj.publish(self.channel, msg.encode())


def run_control_loop():
    ##### Create interface for communication with the firmware #####
    config_file_path = os.path.join(
        get_package_share_directory("robot_fingers"), "config", "trifingeredu.yml"
    )

    robot_data = robot_interfaces.trifinger.SingleProcessData()

    # The backend takes care of communication with the robot hardware.
    robot_backend = robot_fingers.create_trifinger_backend(robot_data, config_file_path)

    # The frontend is used by the user to get observations and send actions
    robot_frontend = robot_interfaces.trifinger.Frontend(robot_data)

    ##### Create LCM interface to communicate with the OSC controller #####
    lc = lcm.LCM()
    input_subscriber = TrifingerInputSubscriber(lc, "TRIFINGER_INPUT")
    state_publisher = TrifingerStatePublisher(lc, "TRIFINGER_STATE")
    timeout = 1e-4  # timeout waiting for incoming lcm message for trifinger inputs

    # Initializes the robot (e.g. performs homing).
    robot_backend.initialize()

    # Because we don't know the current state at beginning
    # to compute desired torque, so it is safe to just send
    # zero torque and obtain observation. Mark this flag to
    # False if not sending the first action.
    is_first_action = True

    while True:
        if is_first_action:
            desired_torque = np.zeros(9)
            is_first_action = False
        else:
            # waiting for new incoming input torque with timeout
            rfds, wfds, efds = select.select([lc.fileno()], [], [], timeout)
            if rfds:
                lc.handle()
            desired_torque = input_subscriber.get_input()

        action = robot_interfaces.trifinger.Action(torque=desired_torque)
        t = robot_frontend.append_desired_action(action)
        robot_frontend.wait_until_timeindex(t)

        cur_observation = robot_frontend.get_observation(t)
        state_publisher.pub_observation(cur_observation)


if __name__ == "__main__":
    run_control_loop()
