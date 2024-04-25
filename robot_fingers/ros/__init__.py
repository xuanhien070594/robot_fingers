"""ROS-related classes/functions."""

import rclpy
import rclpy.node
import rclpy.qos
import robot_interfaces
from threading import Lock
from std_msgs.msg import String
from std_srvs.srv import Empty
from trifinger_msgs.msg import TrifingerState, TrifingerAction
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


class NotificationNode(rclpy.node.Node):
    """Simple ROS node for communication with other processes."""

    _status_topic_name = "~/status"
    _shutdown_service_name = "~/shutdown"

    def __init__(self, name):
        super().__init__(name)

        # quality of service profile keep the last published message for late
        # subscribers (like "latched" in ROS 1).
        qos_profile = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )

        self._status_publisher = self.create_publisher(
            String, self._status_topic_name, qos_profile
        )

        self.shutdown_requested = False
        self._shutdown_srv = self.create_service(
            Empty, self._shutdown_service_name, self.shutdown_callback
        )

    def shutdown_callback(self, request, response):
        self.get_logger().info(
            "Node {} received shutdown request.".format(self.get_name())
        )
        self.shutdown_requested = True
        return response

    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self._status_publisher.publish(msg)


class TrifingerActionSubscriberStatePublisher(rclpy.node.Node):
    def __init__(self, robot_frontend):
        super().__init__("trifinger_action_subscriber_state_publisher")
        self.robot_frontend = robot_frontend

        state_publish_cb_group = MutuallyExclusiveCallbackGroup()
        action_subscribe_cb_group = MutuallyExclusiveCallbackGroup()

        self.publisher_ = self.create_publisher(
            TrifingerState, "/trifinger/joint_states", 10
        )
        timer_period = 1e-3  # seconds
        self.timer = self.create_timer(
            timer_period, self.state_pub_callback, callback_group=state_publish_cb_group
        )

        self.subscription = self.create_subscription(
            TrifingerAction,
            "/trifinger/actions",
            self.action_sub_callback,
            10,
            callback_group=action_subscribe_cb_group,
        )
        self.lock = Lock()
        self._torque = [0.0, 0.0, 0.0]

    def state_pub_callback(self):
        action = robot_interfaces.finger.Action(torque=self._torque)
        t = self.robot_frontend.append_desired_action(action)
        observation = self.robot_frontend.get_observation(t)

        msg = TrifingerState()
        msg.position[:3] = observation.position[:3]
        msg.velocity[:3] = observation.velocity[:3]
        msg.torque[:3] = observation.torque[:3]
        msg.tip_force[:3] = observation.tip_force[:3]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)

    def action_sub_callback(self, msg):
        self.lock.acquire()
        self.torque_ = msg.torque[:3]
        self.lock.release()
