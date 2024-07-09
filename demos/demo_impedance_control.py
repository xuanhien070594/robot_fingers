#!/usr/bin/env python3
"""Basic demo on how to run a Finger Robot with torque control."""
import os
import sys
import datetime
import numpy as np
from ament_index_python.packages import get_package_share_directory

import robot_interfaces
import robot_fingers
import pinocchio
from pinocchio.visualize import MeshcatVisualizer

FINGER_NAME_TO_INDEX_MAPPING = {
    "finger_tip_link_0": 0,
    "finger_tip_link_120": 3,
    "finger_tip_link_240": 6,
}


class ImpedanceController:
    def __init__(self, kp: np.ndarray, kd: np.ndarray, enable_visualizer: bool = False):
        self.kp = kp
        self.kd = kd

        # load trifinger urdf
        urdf_pkg_path = get_package_share_directory("robot_properties_fingers")
        urdf_path = os.path.join(urdf_pkg_path, "urdf/edu", "trifingeredu.urdf")
        self.model, self.collision_model, self.visual_model = (
            pinocchio.buildModelsFromUrdf(urdf_path)
        )
        self.data = self.model.createData()
        self.fingertip_0_frame_id = self.model.getFrameId("finger_tip_link_0")
        self.fingertip_120_frame_id = self.model.getFrameId("finger_tip_link_120")
        self.fingertip_240_frame_id = self.model.getFrameId("finger_tip_link_240")

        # create visualizer if needed
        self.enable_visualizer = enable_visualizer
        if enable_visualizer:
            self.viz = MeshcatVisualizer(
                self.model, self.collision_model, self.visual_model
            )
            try:
                self.viz.initViewer(open=True)
            except ImportError as err:
                print(
                    "Error while initializing the viewer. It seems you should install Python meshcat"
                )
                print(err)
                sys.exit(0)

            # Load the robot in the viewer.
            self.viz.loadViewerModel()

            # Display a robot configuration.
            q0 = pinocchio.neutral(self.model)
            self.viz.display(q0)
            self.viz.displayVisuals(True)
        self.is_new_target = True
        self.fingertip_delta_pos = np.array([0, 0, 0])
        self.log_fingertip_desired_pos = []
        self.log_fingertip_cur_pos = []
        self.log_commanded_torque = []
        self.log_actual_torque = []
        self.log_timestamp = []

    def calc_trifinger_commanded_torque(self, q, dq):
        # compute the current fingertip positions
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        if self.is_new_target:
            self.fingertip_0_desired_target = (
                self.data.oMf[self.fingertip_0_frame_id].translation
                + self.fingertip_delta_pos
            )
            self.fingertip_120_desired_target = (
                self.data.oMf[self.fingertip_120_frame_id].translation
                + self.fingertip_delta_pos
            )
            self.fingertip_240_desired_target = (
                self.data.oMf[self.fingertip_240_frame_id].translation
                + self.fingertip_delta_pos
            )
            self.is_new_target = False

        self.log_fingertip_cur_pos.append(
            self.data.oMf[self.fingertip_0_frame_id].translation
        )
        self.log_fingertip_desired_pos.append(self.fingertip_0_desired_target)

        pinocchio.computeAllTerms(self.model, self.data, q, dq)
        mass_matrix = self.data.M
        nle_term = self.data.g  # only gravity

        commanded_torque_finger_0 = self.calc_commanded_torque_single_finger(
            "finger_tip_link_0",
            mass_matrix,
            q,
            dq,
            self.fingertip_0_desired_target,
        )
        commanded_torque_finger_120 = self.calc_commanded_torque_single_finger(
            "finger_tip_link_120",
            mass_matrix,
            q,
            dq,
            self.fingertip_120_desired_target,
        )
        commanded_torque_finger_240 = self.calc_commanded_torque_single_finger(
            "finger_tip_link_240",
            mass_matrix,
            q,
            dq,
            self.fingertip_240_desired_target,
        )
        commanded_torque = np.concatenate(
            [
                commanded_torque_finger_0,
                commanded_torque_finger_120,
                commanded_torque_finger_240,
            ],
        )
        commanded_torque += nle_term
        return commanded_torque

    def calc_commanded_torque_single_finger(
        self, finger_name, mass_matrix, q, dq, desired_target
    ):
        fingertip_frame = self.model.getFrameId(finger_name)
        fingertip_index = FINGER_NAME_TO_INDEX_MAPPING[finger_name]
        finger_mass_matrix = mass_matrix[
            fingertip_index : (fingertip_index + 3),
            fingertip_index : (fingertip_index + 3),
        ]
        J = pinocchio.computeFrameJacobian(
            self.model, self.data, q, fingertip_frame, pinocchio.ReferenceFrame.WORLD
        )[:3, fingertip_index : (fingertip_index + 3)]
        inv_finger_mass_matrix = np.linalg.inv(finger_mass_matrix)
        effective_mass_matrix = np.linalg.inv(J @ inv_finger_mass_matrix @ J.T)

        pinocchio.updateFramePlacements(self.model, self.data)
        cur_fingertip_pos = self.data.oMf[fingertip_frame].translation
        fingertip_delta_pos = desired_target - cur_fingertip_pos
        fingertip_delta_vel = -J @ dq[fingertip_index : (fingertip_index + 3)]

        commanded_torque = (
            J.T
            @ effective_mass_matrix
            @ (self.kp @ fingertip_delta_pos + self.kd @ fingertip_delta_vel)
        )
        return commanded_torque


def demo_torque_control():
    # move fingertip to follow a square
    count_to_fingertip_delta_pos_old = {
        1000: np.array([0.03, 0, 0]),
        2000: np.array([0.03, 0, 0]),
        3000: np.array([0, 0.03, 0]),
        4000: np.array([0, 0.03, 0]),
        5000: np.array([-0.03, 0.0, 0]),
        6000: np.array([-0.03, 0, 0]),
        7000: np.array([0, -0.03, 0]),
        8000: np.array([0, -0.03, 0]),
        9000: np.array([0, 0, 0.03]),
        10000: np.array([0, 0, 0.03]),
    }

    count_to_fingertip_delta_pos = {
        100: np.array([0.01, 0, 0]),
        200: np.array([0.01, 0, 0]),
        300: np.array([0.01, 0, 0]),
        400: np.array([0.01, 0, 0]),
        500: np.array([0.01, 0, 0]),
        600: np.array([0.01, 0, 0]),
        700: np.array([0, 0.01, 0]),
        800: np.array([0, 0.01, 0]),
        900: np.array([0, 0.01, 0]),
        1000: np.array([0, 0.01, 0]),
        1100: np.array([0, 0.01, 0]),
        1200: np.array([0, 0.01, 0]),
        1300: np.array([-0.01, 0, 0]),
        1400: np.array([-0.01, 0, 0]),
        1500: np.array([-0.01, 0, 0]),
        1600: np.array([-0.01, 0, 0]),
        1700: np.array([-0.01, 0, 0]),
        1800: np.array([-0.01, 0, 0]),
        1900: np.array([0, -0.01, 0]),
        2000: np.array([0, -0.01, 0]),
        2100: np.array([0, -0.01, 0]),
        2200: np.array([0, -0.01, 0]),
        2300: np.array([0, -0.01, 0]),
        2400: np.array([0, -0.01, 0]),
    }

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

    # Initialize impedance controller
    kp = np.diag([1000, 1000, 2000])
    kd = np.diag([13, 13, 13])

    controller = ImpedanceController(kp, kd)

    # Initializes the robot (e.g. performs homing).
    robot_backend.initialize()

    # Because we don't know the current state at beginning
    # to compute desired torque, so it is safe to just send
    # zero torque and obtain observation. Mark this flag to
    # False if not sending the first action.
    is_first_action = True

    action_count = 0

    while action_count < 5000:
        if action_count in count_to_fingertip_delta_pos.keys():
            controller.is_new_target = True
            controller.fingertip_delta_pos = count_to_fingertip_delta_pos[action_count]
        if is_first_action:
            desired_torque = np.zeros(3)
            is_first_action = False
        else:
            desired_torque = controller.calc_trifinger_commanded_torque(
                np.tile(cur_position, 3), np.tile(cur_velocity, 3)
            )[:3]
        action = robot_interfaces.finger.Action(torque=desired_torque)
        t = robot_frontend.append_desired_action(action)
        robot_frontend.wait_until_timeindex(t)

        cur_observation = robot_frontend.get_observation(t)
        cur_position = cur_observation.position
        cur_velocity = cur_observation.velocity

        controller.log_commanded_torque.append(desired_torque)
        controller.log_actual_torque.append(cur_observation.torque)
        controller.log_timestamp.append(action_count)

        action_count += 1

    cur_datetime = datetime.datetime.now().strftime("%d-%b-%Y_%H:%M:%S")
    data = {
        "fingertip_desired_pos": np.array(controller.log_fingertip_desired_pos),
        "fingertip_cur_pos": np.array(controller.log_fingertip_cur_pos),
        "commanded_torque": np.array(controller.log_commanded_torque),
        "actual_torque": np.array(controller.log_actual_torque),
        "timestamp": np.array(controller.log_timestamp),
    }
    np.save(
        f"finger0_PD_follow_square_{cur_datetime}_no_vel_damping",
        data,
    )


if __name__ == "__main__":
    demo_torque_control()
