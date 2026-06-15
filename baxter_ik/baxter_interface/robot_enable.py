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

import errno
import re
import sys
from threading import Lock

import rclpy
from std_msgs.msg import Bool, Empty
import baxter_dataflow

from baxter_core_msgs.msg import AssemblyState
from baxter_interface import settings

class RobotEnable(object):
    param_lock = Lock()

    def __init__(self, node, versioned=False):
        self._node = node
        self._state = None
        state_topic = 'robot/state'
        self._state_sub = self._node.create_subscription(
            AssemblyState,
            state_topic,
            self._state_callback,
            10
            )
            
        if versioned and not self.version_check():
            sys.exit(1)

        baxter_dataflow.wait_for(
            lambda: not self._state is None,
            timeout=5.0,
            timeout_msg=("Failed to get robot state on %s" % (state_topic,)),
            node=self._node
        )

    def _state_callback(self, msg):
        self._state = msg

    def _toggle_enabled(self, status):
        pub = self._node.create_publisher(Bool, 'robot/set_super_enable', 10)
        msg = Bool()
        msg.data = status

        baxter_dataflow.wait_for(
            test=lambda: self._state.enabled == status,
            timeout=2.0 if status else 5.0,
            timeout_msg=("Failed to %sable robot" % ('en' if status else 'dis',)),
            body=lambda: pub.publish(msg),
        )
        self._node.get_logger().info("Robot %s" % ('Enabled' if status else 'Disabled'))

    def state(self):
        return self._state

    def enable(self):
        if self._state.stopped:
            self._node.get_logger().info("Robot Stopped: Attempting Reset...")
            self.reset()
        self._toggle_enabled(True)

    def disable(self):
        self._toggle_enabled(False)

    def reset(self):
        is_reset = lambda: (self._state.enabled == False and
                            self._state.stopped == False and
                            self._state.error == False and
                            self._state.estop_button == 0 and
                            self._state.estop_source == 0)
                            
        pub = self._node.create_publisher(Empty, 'robot/set_super_reset', 10)
        msg = Empty()

        if (self._state.stopped and self._state.estop_button == AssemblyState.ESTOP_BUTTON_PRESSED):
            self._node.get_logger().fatal("E-Stop is ASSERTED.")
            raise IOError(errno.EREMOTEIO, "Failed to Reset: E-Stop Engaged")

        self._node.get_logger().info("Resetting robot...")
        try:
            baxter_dataflow.wait_for(
                test=is_reset,
                timeout=3.0,
                timeout_msg="Failed to reset robot.",
                body=lambda: pub.publish(msg)
            )
        except OSError as e:
            if e.errno == errno.ETIMEDOUT:
                if self._state.error == True and self._state.stopped == False:
                    self._node.get_logger().warn("Non-fatal Robot Error on reset.")
                    return False
            raise

    def stop(self):
        pub = self._node.create_publisher(Empty, 'robot/set_super_stop', 10)
        msg = Empty()
        baxter_dataflow.wait_for(
            test=lambda: self._state.stopped == True,
            timeout=3.0,
            timeout_msg="Failed to stop the robot",
            body=lambda: pub.publish(msg),
        )

    def version_check(self):
        param_name = "rethink.software_version"
        sdk_version = settings.SDK_VERSION

        with self.__class__.param_lock:
            if not self._node.has_parameter(param_name):
                self._node.declare_parameter(param_name, "")
            robot_version = self._node.get_parameter(param_name).value
            
        if not robot_version:
            self._node.get_logger().warn("RobotEnable: Failed to retrieve robot version")
            return False
        else:
            pattern = ("^([0-9]+)\.([0-9]+)\.([0-9]+)")
            match = re.search(pattern, robot_version)
            if not match:
                self._node.get_logger().warn("RobotEnable: Invalid robot version: %s" % robot_version)
                return False
            robot_version = match.string[match.start(1):match.end(3)]
            if robot_version not in settings.VERSIONS_SDK2ROBOT[sdk_version]:
                self._node.get_logger().error("Software Version Mismatch.")
                return False
        return True