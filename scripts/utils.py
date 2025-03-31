from typing import List
from dataclasses import dataclass, field
import math, numpy as np
from math import sqrt, sin, cos, atan, atan2
from functools import singledispatch

PI = 3.1415926535897932384


@dataclass
class State:
    """This dataclass represents the system state (pos and vel)"""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    x_dot: float = 0.0
    y_dot: float = 0.0
    theta_dot: float = 0.0


@dataclass
class Controls:
    """This dataclass represents the system controls"""

    v: float = 0.0
    w: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class GamepadCmds:
    """This dataclass represents the gamepad commands"""

    base_vx: int = 0
    base_vy: int = 0
    base_w: int = 0
    arm_vx: int = 0
    arm_vy: int = 0
    arm_vz: int = 0
    arm_j1: int = 0
    arm_j2: int = 0
    arm_j3: int = 0
    arm_j4: int = 0
    arm_j5: int = 0
    arm_ee: int = 0
    arm_home: int = 0


def print_dataclass(obj):
    print("------------------------------------")
    for field in obj.__dataclass_fields__:
        print(f"{field}: {round(getattr(obj, field), 3)}")
    print("------------------------------------ \n")


class EndEffector:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotx: float = 0.0
    roty: float = 0.0
    rotz: float = 0.0


class FiveDOFRobot:
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """

    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        # self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.155, 0.099, 0.095, 0.055, 0.105

        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0]

        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi / 3, np.pi],
            [-np.pi + np.pi / 12, np.pi - np.pi / 4],
            [-np.pi + np.pi / 12, np.pi - np.pi / 12],
            [-np.pi, np.pi],
        ]

        # End-effector object
        self.ee = EndEffector()

        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices

        self.H05 = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )  # Denavit-Hartenberg parameters (theta, d, a, alpha)
        # Transformation matrices

        self.H_01 = np.empty((4, 4))
        self.H_12 = np.empty((4, 4))
        self.H_23 = np.empty((4, 4))
        self.H_34 = np.empty((4, 4))
        self.H_45 = np.empty((4, 4))

        self.T = np.zeros((self.num_dof, 4, 4))
        # print(self.T)

        ########################################

        ########################################

    def jacobian_v(self):
        """
        blah blah


        """
        # if theta is None:
        # theta = self.theta

        # Initialize a 3x5 matrix for jacobian
        # for first col, take offset vector from H05, and put in first col of matrix
        # for next col, calc offset to be H05 offset - offset between H01
        # continue doing this for each offset
        # combine all collumns into jacobian matrix
        # return jacobian

        self.H_01 = dh_to_matrix([self.theta[0], self.l1, 0, -90])
        self.H_12 = dh_to_matrix([self.theta[1] -90, 0, self.l2, 180])
        self.H_23 = dh_to_matrix([self.theta[2], 0, self.l3, 180])
        self.H_34 = dh_to_matrix([self.theta[3] + 90, 0, 0, 90])
        self.H_45 = dh_to_matrix([self.theta[4], self.l4 + self.l5, 0, 0])

        self.H05 = self.H_01 @ self.H_12 @ self.H_23 @ self.H_34 @ self.H_45

        J = np.empty((3, 5))

        t1 = self.H05[0:3, 3]
        r1 = np.array([0, 0, 1])
        j1 = np.cross(r1, t1)
        J[0:3, 0] = j1

        t2 = self.H05[0:3, 3] - self.H_01[0:3, 3]
        r2 = self.H_01[0:3, 2]
        j2 = np.cross(r2, t2)
        J[0:3, 1] = j2

        t3 = self.H05[0:3, 3] - (self.H_01 @ self.H_12)[0:3, 3]
        r3 = (self.H_01 @ self.H_12)[0:3, 2]
        j3 = np.cross(r3, t3)
        J[0:3, 2] = j3

        t4 = self.H05[0:3, 3] - (self.H_01 @ self.H_12 @ self.H_23)[0:3, 3]
        r4 = (self.H_01 @ self.H_12 @ self.H_23)[0:3, 2]
        j4 = np.cross(r4, t4)
        J[0:3, 3] = j4

        t5 = self.H05[0:3, 3] - (self.H_01 @ self.H_12 @ self.H_23 @ self.H_34)[0:3, 3]
        r5 = (self.H_01 @ self.H_12 @ self.H_23 @ self.H_34)[0:3, 2]
        j5 = np.cross(r5, t5)
        J[0:3, 4] = j5
        # J = np.zeros((5, 3))
        # DH = [
        #     [self.theta[0], self.l1, 0, -90],
        #     [self.theta[1] - 90, 0, self.l2, 180],
        #     [self.theta[2], 0, self.l3, 180],
        #     [self.theta[3] + 90, 0, 0, 90],
        #     [self.theta[4], self.l4 + self.l5, 0, 0],
        # ]

        # T = np.stack(
        #     [
        #         dh_to_matrix(DH[0]),
        #         dh_to_matrix(DH[1]),
        #         dh_to_matrix(DH[2]),
        #         dh_to_matrix(DH[3]),
        #         dh_to_matrix(DH[4]),
        #     ],
        #     axis=0,
        # )


        # T_cumulative = [np.eye(4)]
        # for i in range(5):
        #     T_cumulative.append(T_cumulative[-1] @ T[i])

        # d = T_cumulative[-1] @ np.vstack([0, 0, 0, 1])

        # # Calculate the robot points by applying the cumulative transformations
        # for i in range(0, 5):
        #     T_i = T_cumulative[i]
        #     z = T_i @ np.vstack([0, 0, 1, 0])
        #     d1 = T_i @ np.vstack([0, 0, 0, 1])
        #     r = np.array([d[0] - d1[0], d[1] - d1[1], d[2] - d1[2]]).flatten()
        #     J[i] = np.cross(z[:3].flatten(), r.flatten())

        # offset_12 = self.H_12[0:3, 3]
        # r3 = r2 - offset_12
        # J[0:3, 2] = r3

        # offset_23 = self.H_23[0:3, 3]
        # r4 = r3 - offset_23
        # J[0:3, 1] = r4

        # offset_34 = self.H_34[0:3, 3]
        # r5 = r4 - offset_34
        # J[0:3, 0] = r5

        return J

    def inverse_jacobian(self):
        """
        Creates the inverse jacobian matrix based on the jacobian.

        Returns:
            the pseudo inverse of the jacobian matrix
        """
        J = self.jacobian_v()

        # Calculate pinv of the jacobian
        lambda_constant = 0.01
        J_inv = np.transpose(J) @ np.linalg.inv(
            ((J @ np.transpose(J)) + lambda_constant**2 * np.identity(3))
        )

        return J_inv


def rotm_to_euler(R) -> tuple:
    """Converts a rotation matrix to Euler angles (roll, pitch, yaw).

    Args:
        R (np.ndarray): A 3x3 rotation matrix.

    Returns:
        tuple: Roll, pitch, and yaw angles (in radians).

    Reference:
        Based on the method described at:
        https://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/_modules/klampt/math/so3.html
    """
    r31 = R[2, 0]  # -sin(p)
    r11 = R[0, 0]  # cos(r)*cos(p)
    r33 = R[2, 2]  # cos(p)*cos(y)
    r12 = R[0, 1]  # -sin(r)*cos(y) + cos(r)*sin(p)*sin(y)

    # compute pitch
    # condition r31 to the range of asin [-1, 1]
    r31 = min(1.0, max(r31, -1.0))
    p = -math.asin(r31)
    cosp = math.cos(p)

    if abs(cosp) > 1e-7:
        cosr = r11 / cosp
        # condition cosr to the range of acos [-1, 1]
        cosr = min(1.0, max(cosr, -1.0))
        r = math.acos(cosr)

        cosy = r33 / cosp
        # condition cosy to the range of acos [-1, 1]
        cosy = min(1.0, max(cosy, -1.0))
        y = math.acos(cosy)

    else:
        # pitch (p) is close to 90 deg, i.e. cos(p) = 0.0
        # there are an infinitely many solutions, so we set y = 0
        y = 0
        # r12: -sin(r)*cos(y) + cos(r)*sin(p)*sin(y) -> -sin(r)
        # condition r12 to the range of asin [-1, 1]
        r12 = min(1.0, max(r12, -1.0))
        r = -math.asin(r12)

    r11 = R[0, 0] if abs(R[0, 0]) > 1e-7 else 0.0
    r21 = R[1, 0] if abs(R[1, 0]) > 1e-7 else 0.0
    r32 = R[2, 1] if abs(R[2, 1]) > 1e-7 else 0.0
    r33 = R[2, 2] if abs(R[2, 2]) > 1e-7 else 0.0
    r31 = R[2, 0] if abs(R[2, 0]) > 1e-7 else 0.0

    # print(f"R : {R}")

    if r32 == r33 == 0.0:
        # print("special case")
        # pitch is close to 90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we set yaw = 0
        pitch, yaw = PI / 2, 0.0
        # r12: -sin(r)*cos(y) + cos(r)*sin(p)*sin(y) -> -sin(r)
        # condition r12 to the range of asin [-1, 1]
        r12 = min(1.0, max(r12, -1.0))
        roll = -math.asin(r12)
    else:
        yaw = math.atan2(r32, r33)
        roll = math.atan2(r21, r11)
        denom = math.sqrt(r11**2 + r21**2)
        pitch = math.atan2(-r31, denom)

    return roll, pitch, yaw


def dh_to_matrix(dh_params: list) -> np.ndarray:
    """Converts Denavit-Hartenberg parameters to a transformation matrix.

    Args:
        dh_params (list): Denavit-Hartenberg parameters [theta, d, a, alpha].

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    theta, d, a, alpha = dh_params
    theta = np.radians(theta)
    alpha = np.radians(alpha)
    return np.array(
        [
            [
                cos(theta),
                -sin(theta) * cos(alpha),
                sin(theta) * sin(alpha),
                a * cos(theta),
            ],
            [
                sin(theta),
                cos(theta) * cos(alpha),
                -cos(theta) * sin(alpha),
                a * sin(theta),
            ],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


def euler_to_rotm(rpy: tuple) -> np.ndarray:
    """Converts Euler angles (roll, pitch, yaw) to a rotation matrix.

    Args:
        rpy (tuple): A tuple of Euler angles (roll, pitch, yaw).

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rpy[2]), -math.sin(rpy[2])],
            [0, math.sin(rpy[2]), math.cos(rpy[2])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(rpy[1]), 0, math.sin(rpy[1])],
            [0, 1, 0],
            [-math.sin(rpy[1]), 0, math.cos(rpy[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(rpy[0]), -math.sin(rpy[0]), 0],
            [math.sin(rpy[0]), math.cos(rpy[0]), 0],
            [0, 0, 1],
        ]
    )
    return R_z @ R_y @ R_x


@dataclass
class SimData:
    """Captures simulation data for storage.

    Attributes:
        x (List[float]): x-coordinates over time.
        y (List[float]): y-coordinates over time.
        theta (List[float]): Angles over time.
        x_dot (List[float]): x-velocity over time.
        y_dot (List[float]): y-velocity over time.
        theta_dot (List[float]): Angular velocity over time.
        v (List[float]): Linear velocity over time.
        w (List[float]): Angular velocity over time.
        vx (List[float]): x-component of linear velocity over time.
        vy (List[float]): y-component of linear velocity over time.
    """

    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)
    x_dot: List[float] = field(default_factory=list)
    y_dot: List[float] = field(default_factory=list)
    theta_dot: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)
    vx: List[float] = field(default_factory=list)
    vy: List[float] = field(default_factory=list)


def check_joint_limits(theta: List[float], theta_limits: List[List[float]]) -> bool:
    """Checks if the joint angles are within the specified limits.

    Args:
        theta (List[float]): Current joint angles.
        theta_limits (List[List[float]]): Joint limits for each joint.

    Returns:
        bool: True if all joint angles are within limits, False otherwise.
    """
    for i, th in enumerate(theta):
        if not (theta_limits[i][0] <= th <= theta_limits[i][1]):
            return False
    return True


def calc_distance(p1: State, p2: State) -> float:
    """Calculates the Euclidean distance between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calc_heading(p1: State, p2: State) -> float:
    """Calculates the heading (angle) between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The heading angle in radians.
    """
    return atan2(p1.y - p2.y, p1.x - p2.x)


@singledispatch
def calc_angdiff(p1: State, p2: State) -> float:
    """Calculates the angular difference between two states.

    Args:
        p1 (State): The first state.
        p2 (State): The second state.

    Returns:
        float: The angular difference in radians.
    """
    d = p1.theta - p2.theta
    return math.fmod(d, 2 * math.pi)


@calc_angdiff.register
def _(th1: float, th2: float) -> float:
    """Calculates the angular difference between two angles.

    Args:
        th1 (float): The first angle.
        th2 (float): The second angle.

    Returns:
        float: The angular difference in radians.
    """
    return math.fmod(th1 - th2, 2 * math.pi)


def near_zero(arr: np.ndarray) -> np.ndarray:
    """Checks if elements of an array are near zero.

    Args:
        arr (np.ndarray): The input array.

    Returns:
        np.ndarray: An array with zeros where values are near zero, otherwise the original values.
    """
    tol = 1e-6
    return np.where(np.isclose(arr, 0, atol=tol), 0, arr)
