from distutils.cmd import Command
import math
from time import sleep
from turtle import position

import numpy as np

from geometry_msgs.msg import Twist, Pose, Point, Vector3
from sensor_msgs.msg import Imu
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA, Float32
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaControl
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from datetime import datetime

LIST_PATH = '/home/gmsie/main_ws/src/carla_testbench/config/path.csv'
LOOKAHEAD = 1.0
USING_ODOMETRY_TOPIC = True

GAMMA_0 = 1.0
PHI = 1.0
GAMMA_P = 1.0
LAMBDA = 0.1
KPV = 0.5
KDV = 0.5
# MAX_STEERING = 1.2217
MAX_STEERING = 1.3
V_TRESHOLD = 0.001

A11 = 2.0
A12 = 0.2
A0 = 1.0

EXECUTION_TIME = datetime.today().strftime('%Y-%m-%dT%H:%M:%S')
FILENAME = f'L={LOOKAHEAD:.1f}_{A11=:.1f}_{A12=:.1f}_{A0=:.1f}_{PHI=:.2f}_{GAMMA_0=:.2f}_{EXECUTION_TIME}.csv'

def convert_to_str(data):
    return [str(datum) for datum in data]

def log_output(*data):
    with open(FILENAME, 'a') as output_file:
        output_file.write(','.join(convert_to_str(data)))
        output_file.write('\n')

def triangle_height(A, B1, B2):
    b = np.hypot(*(B2-B1))
    l1 = np.hypot(*(A-B1))
    l2 = np.hypot(*(A-B2))

    cos_a = (l1**2 + b**2 - l2**2)/(2*l1*b)
    sin_a = np.sqrt(1 - cos_a**2)

    h = l1*sin_a

    return h

def get_curvature(A, B, C):
    l = np.hypot(*(B-C))
    h = triangle_height(A, B, C)

    area = h*l/2

    curvature = 4*area/(np.hypot(*(A-B))*np.hypot(*(A-B))*np.hypot(*(B-C)))

    if np.isnan(curvature):
        return 0.0

    return curvature

def path_min_distance(A, B1, B2, B3):
    d1 = triangle_height(A, B1, B2)
    d2 = triangle_height(A, B2, B3)

    return np.min([d1, d2])

def euler_from_quaternion(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """

        x = q.x
        y = q.y
        z = q.z
        w = q.w

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

class FrameListener(Node):

    def __init__(self):
        super().__init__('sliding_modes_controller')

        self.path = np.genfromtxt(LIST_PATH, delimiter=',')
        self.path_index = 0

        # Declare and acquire `target_frame` parameter
        self.declare_parameter('target_frame', 'ego_vehicle')
        self.target_frame = self.get_parameter(
            'target_frame').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 1)
        self.carla_control_pub = self.create_publisher(CarlaControl, '/carla/control', 3)
        self.markers_pub = self.create_publisher(MarkerArray, '/carla/debug_marker', 1)

        self.subscription_odometry = self.create_subscription(
            Odometry,
            '/carla/ego_vehicle/odometry',
            self._receive_odometry,
            1
        )

        self.subscription_imu = self.create_subscription(
            Imu,
            '/carla/ego_vehicle/imu',
            self._receive_imu,
            1
        )

        # Call on_timer function every second
        self.timer = self.create_timer(1.0, self._on_timer)
        self.timer_count = 0

        self.counter = 0
        self.imu_data = None
        self.odometry_data = None

        self.odometry_data = True

        self.received_imu = False
        self.received_odometry = False

        self.last_v_error = 0

        self.carla_control_pub.publish(CarlaControl(command=CarlaControl.PAUSE))
        self.carla_control_pub.publish(CarlaControl(command=CarlaControl.STEP_ONCE))

    def _on_timer(self):
        header = Header(
            # seq=self.counter,
            frame_id='map'
        )

        new_marker_list = [Marker(
            header=header,
            ns='path',
            id=n,
            type=Marker.SPHERE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(
                    x = point[0],
                    y = point[1],
                    z = 0.2
                )
            ),
            scale=Vector3(x=0.4, y=0.4, z=0.4),
            color=ColorRGBA(r=1., a=1.)
        ) for n, point in enumerate(self.path)]

        self.markers_pub.publish(MarkerArray(markers=new_marker_list))

    def _receive_imu(self, msg):
        self.received_imu = True
        self.imu_data = msg
        self._control_action()

    def _receive_odometry(self, msg):
        self.received_odometry = True
        self.odometry_data = msg
        self._control_action()

    def _control_action(self):
        if (not self.received_imu) or (not self.received_odometry):
            return

        _, _, psi = euler_from_quaternion(self.odometry_data.pose.pose.orientation)
        pos_x = self.odometry_data.pose.pose.position.x
        pos_y = self.odometry_data.pose.pose.position.y

        psi_dot = self.odometry_data.twist.twist.angular.z
        v_x = self.odometry_data.twist.twist.linear.x
        v_y = self.odometry_data.twist.twist.linear.y

        a_x = self.imu_data.linear_acceleration.x
        a_y = self.imu_data.linear_acceleration.y

        self.control_callback(psi, pos_x, pos_y, psi_dot, v_x, v_y, a_x, a_y)

        self.received_imu = False
        self.received_odometry = False

    def _gamma(self, beta, psi_dot, v_x, v_y, gamma_0):
        v = np.hypot(v_x, v_y)
        if v < V_TRESHOLD:
            return gamma_0/2
        else:
            # return v*(A11/v*np.abs(beta) + A12/v**2*np.abs(psi_dot) + np.abs(beta)*A0*np.abs()) + gamma_0/2
            return v*(A11/v*np.abs(beta) + A12/v**2*np.abs(psi_dot)) + gamma_0/2

    def get_position(self):
        # Store frame names in variables that will be used to
        # compute transformations
        if USING_ODOMETRY_TOPIC:
            return self.odometry_data.pose.x, self.odometry_data.pose.y, self.odometry_data.pose.z
        else:
            from_frame_rel = self.target_frame
            to_frame_rel = 'map'
            try:
                now = rclpy.time.Time()
                trans = self.tf_buffer.lookup_transform(
                    to_frame_rel,
                    from_frame_rel,
                    now
            )
            except TransformException as ex:
                self.get_logger().info(
                    f'Could not transform "{to_frame_rel}" to "{from_frame_rel}": {ex}')
                return

            return trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z

    def get_velocity(self):
        return self.odometry_data.twist.x, self.odometry_data.twist.y, self.odometry_data.twist.z

    def place_landmark(self, pos_x, pos_y, pos_z, counter):
        header = Header(
            # seq=self.counter,
            frame_id='map'
        )

        new_marker = Marker(
            header=header,
            ns='ego_vehicle',
            id=counter,
            type=Marker.SPHERE,
            action=Marker.ADD,
            pose=Pose(
                position=Point(
                    x = pos_x,
                    y = pos_y,
                    z = pos_z
                )
            ),
            scale=Vector3(x=0.5, y=0.5, z=0.5),
            color=ColorRGBA(b=1., a=1.)
        )

        self.markers_pub.publish(MarkerArray(markers=[new_marker]))

    def remove_landmark(self, counter):
        header = Header(
            # seq=self.counter,
            frame_id='map'
        )

        new_marker = Marker(
            header=header,
            ns='ego_vehicle',
            id=counter,
            action=Marker.DELETE,
        )

        self.markers_pub.publish(MarkerArray(markers=[new_marker]))

    def stop_car(self):
        print('Stop car!')
        for _ in range(10):
            self.cmd_pub.publish(CarlaEgoVehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            sleep(0.1)

    def control_callback(self, psi, pos_x, pos_y, psi_dot, v_x, v_y, a_x, a_y):

        # pos_x, pos_y, pos_z = self.get_position()
        v = np.hypot(v_x, v_y)

        distances = np.hypot(
            self.path.T[0] - pos_x,
            self.path.T[1] - pos_y
        )

        forward_points_distances = distances[self.path_index + 1:]
        start_point = np.argmin(forward_points_distances)

        advance = np.argmax(forward_points_distances[start_point:] > LOOKAHEAD)

        closest = np.argmin(distances)

        if closest == 559:
            raise KeyboardInterrupt

        self.path_index += (advance + start_point)

        # self.get_logger().info(f'Advanced {advance} Start {start_point} Index {self.path_index} X: {pos_x:.3f} Y: {pos_y:.3f} v: {v:.3f}')
        self.get_logger().info(f'Advanced {advance} Start {start_point} Index {self.path_index} Closest {closest} Total {len(distances)}')

        desired_x, desired_y = self.path[self.path_index + 1]

        if advance + start_point > 0:
            self.remove_landmark(self.counter - 1)
            self.place_landmark(desired_x, desired_y, 0.2, self.counter)
            self.counter += 1

        delta_x = desired_x - pos_x
        delta_y = desired_y - pos_y

        beta = 0
        if np.abs(v) > 0.001:
            beta = np.arctan2(v_y, v_x)

        rho = np.min([np.hypot(delta_x, delta_y), LOOKAHEAD])
        desired_psi = np.arctan2(delta_y, delta_x)

        # previous_psi_vector = self.path[closest] - self.path[closest - 1]
        # previous_psi = np.arctan2(previous_psi_vector[1], previous_psi_vector[0])

        # next_psi_vector = self.path[closest + 1] - self.path[closest]
        # next_psi = np.arctan2(next_psi_vector[1], next_psi_vector[0])

        # delta_previous_next_vector = self.path[closest + 1] - self.path[closest - 1]
        # delta_previous_next = np.hypot(delta_previous_next_vector[1], delta_previous_next_vector[0])

        # delta_psi = next_psi - previous_psi
        # self.get_logger().info(f'Psi dot => {de`lta_psi=:.3f} {delta_previous_next=:.3f}')

        # if delta_psi > np.pi:
        #     delta_psi -= 2*np.pi
        # if delta_psi < -np.pi:
        #     delta_psi += 2*np.pi

        curvature = get_curvature(
            self.path[closest - 3],
            self.path[closest],
            self.path[closest + 3]
        )

        # curvature = get_curvature(
        #     self.path[self.path_index - 3],
        #     [pos_x, pos_y],
        #     self.path[self.path_index + 3]
        # )

        desired_psi_dot = v*curvature

        gamma = self._gamma(beta, psi_dot, v_x, v_y, GAMMA_0)

        # psi = self.normalize_angle(psi)

        psi_beta = psi + beta
        if psi_beta > np.pi:
            psi_beta -= 2*np.pi
        if psi_beta < -np.pi:
            psi_beta += 2*np.pi

        delta_psi = psi_beta - desired_psi
        if delta_psi > np.pi:
            delta_psi -= 2*np.pi
        if delta_psi < -np.pi:
            delta_psi += 2*np.pi

        # delta = -( -gamma*np.clip((delta_psi)/PHI, -1, 1) - v*desired_psi_dot)
        delta = -( -gamma*np.clip((delta_psi)/PHI, -1, 1) )
        delta = np.clip(delta, -MAX_STEERING, MAX_STEERING)

        delta_beta = delta + beta
        if delta_beta > np.pi:
            delta_beta -= 2*np.pi
        if delta_beta < -np.pi:
            delta_beta += 2*np.pi

        self.get_logger().info(f'Delta => {gamma=:.3f} {desired_psi=:.3f} {psi=:.3f} {beta=:.3f} {delta_psi=:.3f} {psi_dot=:.3f} {v_x=:.3f} {v_y=:.3f} {curvature=:.3f}')

        # filtered_a_y = (a_y + self.last_m_ay + 5.324*self.last_f_ay)/7.314
        # desired_v = np.max([5.5*rho/LOOKAHEAD - 5.0*np.abs(np.sin(beta)), 0])
        desired_v = 5.5*rho/LOOKAHEAD
        # if curvature > 0.001 :
            # desired_v = np.min([5.5*rho/LOOKAHEAD, 0.2/curvature])

        v_error = desired_v - v
        v_error_dot = v_error - self.last_v_error
        fx = (v_error + v_error_dot*KDV)*KPV
        fx = np.clip(fx, -1, 1)

        self.last_v_error = v_error

        self.get_logger().info(f'{desired_v=:.3f}')
        self.get_logger().info(f'Control action: fx = {fx:.3f} delta = {delta:.3f}')

        self.cmd_pub.publish(CarlaEgoVehicleControl(throttle=np.max([fx, 0]), brake=-np.min([fx, 0]), steer=delta/MAX_STEERING))

        self.carla_control_pub.publish(CarlaControl(command=CarlaControl.STEP_ONCE))

        log_output(
            pos_x,
            pos_y,
            psi,
            v_x,
            v_y,
            psi_dot,
            desired_x,
            desired_y,
            self.path_index,
            closest,
            path_min_distance([pos_x, pos_y], self.path[closest - 1], self.path[closest], self.path[closest + 1]),
            delta,
            fx,
            gamma,
            curvature,
            desired_psi_dot,
            a_x,
            a_y
        )

    def normalize_angle(self, ang):
        ang = ang % 2*np.pi
        if ang > np.pi:
            return ang - 2*np.pi
        if ang < -np.pi:
            return ang + 2*np.pi
        return ang

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = FrameListener()

    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        minimal_publisher.stop_car()
        minimal_publisher.destroy_node()
        rclpy.shutdown()
        print(FILENAME)


if __name__ == '__main__':
    main()
