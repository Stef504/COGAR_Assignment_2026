# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.

import collections
from copy import deepcopy

import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

import baxter_dataflow

from baxter_core_msgs.msg import JointCommand, EndpointState
from baxter_interface import settings

class Limb(object):
    Point = collections.namedtuple('Point', ['x', 'y', 'z'])
    Quaternion = collections.namedtuple('Quaternion', ['x', 'y', 'z', 'w'])

    def __init__(self, node, limb):
        self._node = node
        self.name = limb
        self._joint_angle = dict()
        self._joint_velocity = dict()
        self._joint_effort = dict()
        self._cartesian_pose = dict()
        self._cartesian_velocity = dict()
        self._cartesian_effort = dict()

        self._joint_names = {
            'left': ['left_s0', 'left_s1', 'left_e0', 'left_e1',
                     'left_w0', 'left_w1', 'left_w2'],
            'right': ['right_s0', 'right_s1', 'right_e0', 'right_e1',
                      'right_w0', 'right_w1', 'right_w2']
            }

        ns = '/robot/limb/' + limb + '/'
        self._command_msg = JointCommand()
        latch_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._pub_speed_ratio = self._node.create_publisher(
            Float64, ns + 'set_speed_ratio', latch_qos)

        self._pub_joint_cmd = self._node.create_publisher(
            JointCommand, ns + 'joint_command', 10)

        self._pub_joint_cmd_timeout = self._node.create_publisher(
            Float64, ns + 'joint_command_timeout', latch_qos)

        self._cartesian_state_sub = self._node.create_subscription(
            EndpointState, ns + 'endpoint_state', self._on_endpoint_states, 1)

        joint_state_topic = 'robot/joint_states'
        self._joint_state_sub = self._node.create_subscription(
            JointState, joint_state_topic, self._on_joint_states, 1)

        err_msg = ("%s limb init failed to get current joint_states from %s") % (self.name.capitalize(), joint_state_topic)
        baxter_dataflow.wait_for(lambda: len(self._joint_angle.keys()) > 0, timeout_msg=err_msg)
        err_msg = ("%s limb init failed to get current endpoint_state from %s") % (self.name.capitalize(), ns + 'endpoint_state')
        baxter_dataflow.wait_for(lambda: len(self._cartesian_pose.keys()) > 0, timeout_msg=err_msg)

    def _on_joint_states(self, msg):
        for idx, name in enumerate(msg.name):
            if name in self._joint_names[self.name]:
                self._joint_angle[name] = msg.position[idx]
                self._joint_velocity[name] = msg.velocity[idx]
                self._joint_effort[name] = msg.effort[idx]

    def _on_endpoint_states(self, msg):
        self._cartesian_pose = {
            'position': self.Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
            'orientation': self.Quaternion(msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w)
        }
        self._cartesian_velocity = {
            'linear': self.Point(msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z),
            'angular': self.Point(msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z)
        }
        self._cartesian_effort = {
            'force': self.Point(msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z),
            'torque': self.Point(msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z)
        }

    def joint_names(self): return self._joint_names[self.name]
    def joint_angles(self): return deepcopy(self._joint_angle)
    
    # ... Other accessors identical in logic, omitted for brevity ...
    def set_command_timeout(self, timeout):
        msg = Float64()
        msg.data = float(timeout)
        self._pub_joint_cmd_timeout.publish(msg)

    def set_joint_positions(self, positions, raw=False):
        self._command_msg.names = list(positions.keys())
        self._command_msg.command = list(positions.values())
        if raw:
            self._command_msg.mode = JointCommand.RAW_POSITION_MODE
        else:
            self._command_msg.mode = JointCommand.POSITION_MODE
        self._pub_joint_cmd.publish(self._command_msg)
        
    def move_to_neutral(self, timeout=15.0):
        angles = dict(zip(self.joint_names(), [0.0, -0.55, 0.0, 0.75, 0.0, 1.26, 0.0]))
        return self.move_to_joint_positions(angles, timeout)

    def move_to_joint_positions(self, positions, timeout=15.0, threshold=settings.JOINT_ANGLE_TOLERANCE, test=None):
        cmd = self.joint_angles()

        def filtered_cmd():
            for joint in positions.keys():
                cmd[joint] = 0.012488 * positions[joint] + 0.98751 * cmd[joint]
            return cmd

        def genf(joint, angle):
            def joint_diff():
                return abs(angle - self._joint_angle[joint])
            return joint_diff

        diffs = [genf(j, a) for j, a in positions.items() if j in self._joint_angle]

        self.set_joint_positions(filtered_cmd())
        baxter_dataflow.wait_for(
            test=lambda: callable(test) and test() == True or (all(diff() < threshold for diff in diffs)),
            timeout=timeout,
            timeout_msg=("%s limb failed to reach commanded joint positions" % (self.name.capitalize(),)),
            rate=100,
            raise_on_error=False,
            body=lambda: self.set_joint_positions(filtered_cmd())
            )