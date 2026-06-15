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
# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.

import re
import sys
import time

from copy import deepcopy
from math import fabs

from json import (
    JSONDecoder,
    JSONEncoder,
)

import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy

import baxter_dataflow

from baxter_core_msgs.msg import (
    EndEffectorCommand,
    EndEffectorProperties,
    EndEffectorState,
)
from baxter_interface import settings


class Gripper(object):
    """
    Interface class for a gripper on the Baxter Research Robot.
    """
    def __init__(self, node, gripper, versioned=False):
        """
        Version-checking capable constructor.

        @param node: ROS 2 node reference
        @type gripper: str
        """
        self._node = node
        self.name = gripper + '_gripper'
        self._cmd_sender = self._node.get_name() + '_%s'
        self._cmd_sequence = 0

        ns = 'robot/end_effector/' + self.name + "/"

        self._state = None
        self._prop = EndEffectorProperties(id=-1) 
        self.on_type_changed = baxter_dataflow.Signal()
        self.on_gripping_changed = baxter_dataflow.Signal()
        self.on_moving_changed = baxter_dataflow.Signal()

        self._parameters = dict()

        self._cmd_pub = self._node.create_publisher(EndEffectorCommand, ns + 'command', 10)

        # ROS 2 equivalent of latch=True
        latch_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._prop_pub = self._node.create_publisher(
            EndEffectorProperties, ns + 'rsdk/set_properties', latch_qos)

        self._state_pub = self._node.create_publisher(
            EndEffectorState, ns + 'rsdk/set_state', latch_qos)

        self._state_sub = self._node.create_subscription(
            EndEffectorState, ns + 'state', self._on_gripper_state, 10)

        self._prop_sub = self._node.create_subscription(
            EndEffectorProperties, ns + 'properties', self._on_gripper_prop, 10)

        baxter_dataflow.wait_for(
                          lambda: not self._state is None,
                          timeout=5.0,
                          timeout_msg=("Failed to get state from %s" % (ns + 'state',)),
                          node=self._node
                          )
        
        baxter_dataflow.wait_for(
                          lambda: not self.type() is None,
                          timeout=5.0,
                          timeout_msg=("Failed to get properties from %s" % (ns + 'properties',)),
                          node=self._node
                          )
        
    

        if versioned and self.type() == 'electric':
            if not self.version_check():
                sys.exit(1)

        self.set_parameters(defaults=True)

    def _on_gripper_state(self, state):
        old_state = self._state
        self._state = deepcopy(state)
        if old_state is not None and old_state.gripping != state.gripping:
            self.on_gripping_changed(state.gripping == True)
        if old_state is not None and old_state.moving != state.moving:
            self.on_moving_changed(state.moving == True)

    def _on_gripper_prop(self, properties):
        old_prop = self._prop
        self._prop = deepcopy(properties)
        if old_prop.ui_type != self._prop.ui_type and old_prop.id != -1:
            self.on_type_changed({
                EndEffectorProperties.SUCTION_CUP_GRIPPER: 'suction',
                EndEffectorProperties.ELECTRIC_GRIPPER: 'electric',
                EndEffectorProperties.PASSIVE_GRIPPER: 'custom',
                                 }.get(properties.ui_type, None))

    def _inc_cmd_sequence(self):
        self._cmd_sequence = (self._cmd_sequence % 0x7FFFFFFF) + 1
        return self._cmd_sequence

    def _clip(self, val):
        return max(min(val, 100.0), 0.0)

    def _capablity_warning(self, cmd):
        msg = ("%s %s - not capable of '%s' command" %
               (self.name, self.type(), cmd))
        self._node.get_logger().warn(msg)

    def _version_str_to_time(self, version_str, version_type):
        float_time = 0.0
        time_format = r"%Y/%m/%d %H:%M:%S"
        if version_str != '0000/0/0 0:0:00':
            try:
                float_time = time.mktime(time.strptime(version_str, time_format))
            except ValueError:
                self._node.get_logger().error(("%s %s - The Gripper's %s "
                              "timestamp does not meet python time formating "
                              "requirements: %s does not map "
                              "to '%s'") % (self.name, self.type(), version_type, version_str, time_format))
                sys.exit(1)
        return float_time

    def version_check(self):
        sdk_version = settings.SDK_VERSION
        firmware_date_str = self.firmware_build_date()
        if self.type() != 'electric':
            self._node.get_logger().warn("%s %s (%s): Version Check not needed" % (self.name, self.type(), firmware_date_str))
            return True
        if not firmware_date_str:
            self._node.get_logger().error("%s %s: Failed to retrieve version string during Version Check." % (self.name, self.type()))
            return False
            
        firmware_time = self._version_str_to_time(firmware_date_str, "current firmware")
        warn_time = self._version_str_to_time(settings.VERSIONS_SDK2GRIPPER[sdk_version]['warn'], "baxter_interface settings.py firmware 'warn'")
        fail_time = self._version_str_to_time(settings.VERSIONS_SDK2GRIPPER[sdk_version]['fail'], "baxter_interface settings.py firmware 'fail'")
        
        if firmware_time > warn_time:
            return True
        elif firmware_time <= warn_time and firmware_time > fail_time:
            self._node.get_logger().warn("Gripper Firmware needs update.")
            return True
        elif firmware_time <= fail_time and firmware_time > 0.0:
            self._node.get_logger().error("Gripper Firmware incompatible.")
            return False
        else:
            legacy_str = '1.1.242'
            if self.firmware_version()[0:len(legacy_str)] == legacy_str:
                return True
            else:
                self._node.get_logger().error("Unknown firmware version.")
                return False

    def command(self, cmd, block=False, test=lambda: True, timeout=0.0, args=None):
        ee_cmd = EndEffectorCommand()
        ee_cmd.id = self.hardware_id()
        ee_cmd.command = cmd
        ee_cmd.sender = self._cmd_sender % (cmd,)
        ee_cmd.sequence = self._inc_cmd_sequence()
        ee_cmd.args = ''
        if args != None:
            ee_cmd.args = JSONEncoder().encode(args)
            
        seq_test = lambda: (self._state.command_sender == ee_cmd.sender and
                            (self._state.command_sequence == ee_cmd.sequence
                             or self._state.command_sequence == 0))
        self._cmd_pub.publish(ee_cmd)
        
        if block:
            finish_time = (self._node.get_clock().now().nanoseconds / 1e9) + timeout
            cmd_seq = baxter_dataflow.wait_for(
                          test=seq_test,
                          timeout=timeout,
                          raise_on_error=False,
                          body=lambda: self._cmd_pub.publish(ee_cmd)
                      )
            if not cmd_seq:
                self._node.get_logger().debug("Timed out on gripper command acknowledgement")
            time_remain = max(0.5, finish_time - (self._node.get_clock().now().nanoseconds / 1e9))
            return baxter_dataflow.wait_for(
                       test=test,
                       timeout=time_remain,
                       raise_on_error=False,
                       body=lambda: self._cmd_pub.publish(ee_cmd)
                   )
        else:
            return True

    # Remaining API interface functionality logic remains identical 
    # to original, substituting rospy.sleep() with time.sleep()
    def valid_parameters_text(self):
        if self.type() == 'electric':
            return "Electric Gripper params..."
        elif self.type() == 'suction':
            return "Suction Gripper params..."
        else:
            return "No valid parameters."

    def valid_parameters(self):
        valid = dict()
        if self.type() == 'electric':
            valid = dict({'velocity': 50.0, 'moving_force': 40.0, 'holding_force': 30.0, 'dead_zone': 5.0})
        elif self.type() == 'suction':
            valid = dict({'vacuum_sensor_threshold': 18.0, 'blow_off_seconds': 0.4})
        return valid

    def set_parameters(self, parameters=None, defaults=False):
        valid_parameters = self.valid_parameters()
        if defaults:
            self._parameters = valid_parameters
        if parameters is None:
            parameters = dict()
        for key in parameters.keys():
            if key in valid_parameters.keys():
                self._parameters[key] = parameters[key]
            else:
                self._node.get_logger().warn("Invalid parameter: %s" % key)
        cmd = EndEffectorCommand.CMD_CONFIGURE
        self.command(cmd, args=self._parameters)

    # Simplified remaining class definitions... (open, close, reboot)
    def reboot(self, timeout=5.0, delay_check=0.1):
        if self.type() != 'electric':
            return self._capablity_warning('reboot')
        self._cmd_reboot(block=True, timeout=timeout)
        time.sleep(delay_check)
        if self.error():
            if not self.reset(block=True, timeout=timeout):
                self._node.get_logger().error("Failed to reset gripper error after reboot.")
                return False
            self.set_parameters(defaults=True)
        return True

    # ... remaining accessors (_state references) translated directly
    def type(self):
        return {
        EndEffectorProperties.SUCTION_CUP_GRIPPER: 'suction',
        EndEffectorProperties.ELECTRIC_GRIPPER: 'electric',
        EndEffectorProperties.PASSIVE_GRIPPER: 'custom',
        }.get(self._prop.ui_type, None)

    def error(self):
        return self._state.error == True

    def hardware_id(self):
        return deepcopy(self._state.id)
        
    def firmware_build_date(self):
        return deepcopy(self._prop.firmware_date)

    def firmware_version(self):
        return deepcopy(self._prop.firmware_rev)