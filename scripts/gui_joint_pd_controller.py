#!/usr/bin/env python3
import sys
import time
import os

import numpy as np

import robot_interfaces
import robot_fingers

from ament_index_python.packages import get_package_share_directory
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker


class ControlThread(QThread):
    # Signal to communicate updates to the main thread
    update_signal = pyqtSignal()

    def __init__(self, robot_frontend, current_positions, target_positions, mutex):
        super().__init__()
        self.robot_frontend = robot_frontend
        self.current_positions = current_positions
        self.target_positions = target_positions
        self.mutex = mutex

    def run(self):
        while True:
            # Run a position controller that randomly changes the desired position
            # every 500 steps.  One time step corresponds to roughly 1 ms.

            with QMutexLocker(self.mutex):
                desired_position = np.concatenate(
                    [finger_pos for finger_pos in self.current_positions.values()]
                )
                assert all(pos != 0 for pos in desired_position)
            for _ in range(100):
                # Appends a torque command ("action") to the action queue.
                # Returns the time step at which the action is going to be
                # executed.
                action = robot_interfaces.trifinger.Action(position=desired_position)
                t = self.robot_frontend.append_desired_action(action)

                # wait until the action is executed
                self.robot_frontend.wait_until_timeindex(t)

            # print observation of the current time step
            with QMutexLocker(self.mutex):
                cur_position = self.robot_frontend.get_observation(t).position
                self.current_positions["Finger 0"] = cur_position[:3]
                self.current_positions["Finger 1"] = cur_position[3:6]
                self.current_positions["Finger 2"] = cur_position[6:]

            # Emit a signal to update the GUI
            self.update_signal.emit()


class RobotController(QWidget):
    def __init__(self):
        super().__init__()

        # --------------- Initialize GUI ------------------------------------
        # Initial current and target positions for each finger
        self.target_positions = {
            "Finger 0": [0] * 3,
            "Finger 1": [0] * 3,
            "Finger 2": [0] * 3,
        }

        self.current_positions = {
            "Finger 0": [0] * 3,
            "Finger 1": [0] * 3,
            "Finger 2": [0] * 3,
        }

        # Mutex for thread-safe access to target_positions and edit current_positions
        self.mutex = QMutex()

        # Create a layout for each finger section
        self.joint_cur_pos_labels = {}
        self.joint_target_pos_labels = {}

        # Create UI elements
        self.init_ui()

        # --------------- Intialize robot driver ------------------------------
        config_file_path = os.path.join(
            get_package_share_directory("robot_fingers"), "config", "trifingeredu.yml"
        )

        robot_data = robot_interfaces.trifinger.SingleProcessData()

        # The backend takes care of communication with the robot hardware.
        robot_backend = robot_fingers.create_trifinger_backend(
            robot_data, config_file_path
        )

        # The frontend is used by the user to get observations and send actions
        robot_frontend = robot_interfaces.trifinger.Frontend(robot_data)

        # Initializes the robot (e.g. performs homing).
        robot_backend.initialize()

        # Get the first joint positions and set values to GUI
        action = robot_interfaces.finger.Action(torque=np.zeros(9))
        t = robot_frontend.append_desired_action(action)
        robot_frontend.wait_until_timeindex(t)

        cur_joint_positions = robot_frontend.get_observation(t).position
        self.current_positions["Finger 0"] = cur_joint_positions[:3]
        self.current_positions["Finger 1"] = cur_joint_positions[3:6]
        self.current_positions["Finger 2"] = cur_joint_positions[6:]
        self.target_positions["Finger 0"] = cur_joint_positions[:3]
        self.target_positions["Finger 1"] = cur_joint_positions[3:6]
        self.target_positions["Finger 2"] = cur_joint_positions[6:]

        # ------------------------ Start control loop  ------------------------------
        # Start the background thread
        self.worker_thread = ControlThread(
            None, self.current_positions, self.target_positions, self.mutex
        )
        self.worker_thread.update_signal.connect(self.update_current_position)
        self.worker_thread.start()

    def init_ui(self):
        main_layout = QVBoxLayout()

        for finger in ["Finger 0", "Finger 1", "Finger 2"]:
            finger_layout = QVBoxLayout()

            # label to display name of the finger
            finger_label = QLabel(finger)
            finger_layout.addWidget(finger_label)

            for i in range(3):
                joint_layout = QVBoxLayout()
                joint_name_label = QLabel(f"Joint {i}")
                joint_cur_pos_label = QLabel(
                    f"Current Position: {self.current_positions[finger][i]:.4f}"
                )
                joint_cur_pos_label.setStyleSheet("font-weight: bold")
                self.joint_cur_pos_labels[(finger, i)] = joint_cur_pos_label
                joint_layout.addWidget(joint_name_label)
                joint_layout.addWidget(joint_cur_pos_label)

                joint_target_pos_layout = QHBoxLayout()
                joint_target_pos_label = QLabel(
                    f"Target Position: {self.target_positions[finger][i]:.4f}"
                )
                joint_target_pos_label.setStyleSheet("font-weight: bold")
                self.joint_target_pos_labels[(finger, i)] = joint_target_pos_label
                plus_button = QPushButton(f"+")
                minus_button = QPushButton(f"-")
                joint_target_pos_layout.addWidget(joint_target_pos_label)
                joint_target_pos_layout.addWidget(plus_button)
                joint_target_pos_layout.addWidget(minus_button)
                joint_target_pos_layout.addStretch()

                joint_layout.addLayout(joint_target_pos_layout)
                finger_layout.addLayout(joint_layout)

                # Connect buttons to their respective handlers
                plus_button.clicked.connect(
                    lambda checked, index=i, f=finger: self.update_target_position(
                        f, index, 0.01
                    )
                )
                minus_button.clicked.connect(
                    lambda checked, index=i, f=finger: self.update_target_position(
                        f, index, -0.01
                    )
                )
            finger_layout.addStretch()

            # Add the finger layout to the main layout
            main_layout.addLayout(finger_layout)
        main_layout.addStretch()

        # Set main layout
        self.setLayout(main_layout)
        self.setWindowTitle("Robot Target Position Controller")
        self.show()

    def update_target_position(self, finger, index, delta):
        with QMutexLocker(self.mutex):
            self.target_positions[finger][index] += delta
        if (finger, index) in self.joint_target_pos_labels:
            self.joint_target_pos_labels[(finger, index)].setText(
                f"Target Position: {self.target_positions[finger][index]:.4f}"
            )

    def update_current_position(self):
        with QMutexLocker(self.mutex):
            for (finger, index), label in self.joint_cur_pos_labels.items():
                label.setText(
                    f"Current Position: {self.current_positions[finger][index]:.4f}"
                )

    def closeEvent(self, event):
        # Stop the worker thread when closing the application
        self.worker_thread.terminate()
        self.worker_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = RobotController()
    sys.exit(app.exec_())
