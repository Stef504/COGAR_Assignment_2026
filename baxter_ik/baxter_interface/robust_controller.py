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
import rclpy

from baxter_core_msgs.msg import RobustControllerStatus

class RobustController(object):
    (STATE_IDLE,
     STATE_STARTING,
     STATE_RUNNING,
     STATE_STOPPING) = range(4)

    def __init__(self, node, namespace, enable_msg, disable_msg, timeout=60):
        self._node = node
        self._command_pub = self._node.create_publisher(
            type(enable_msg),
            namespace + '/enable',
            10)
            
        self._status_sub = self._node.create_subscription(
            RobustControllerStatus,
            namespace + '/status',
            self._callback,
            10)

        self._enable_msg = enable_msg
        self._disable_msg = disable_msg
        self._timeout = timeout
        self._state = self.STATE_IDLE
        self._return = 0

        # Hook into rclpy shutdown
        rclpy.get_default_context().on_shutdown(self._on_shutdown)

    def _callback(self, msg):
        if self._state == self.STATE_RUNNING:
            if msg.complete == RobustControllerStatus.COMPLETE_W_SUCCESS:
                self._state = self.STATE_STOPPING
                self._return = 0
            elif msg.complete == RobustControllerStatus.COMPLETE_W_FAILURE:
                self._state = self.STATE_STOPPING
                self._return = errno.EIO
            elif not msg.is_enabled:
                self._state = self.STATE_IDLE
                self._return = errno.ENOMSG

        elif self._state == self.STATE_STOPPING and not msg.is_enabled:
            self._state = self.STATE_IDLE

        elif self._state == self.STATE_STARTING and msg.is_enabled:
            self._state = self.STATE_RUNNING

    def _run_loop(self):
        import time
        start = self._node.get_clock().now()

        while rclpy.ok():
            if (self._state == self.STATE_RUNNING and
                (self._node.get_clock().now() - start).nanoseconds / 1e9 > self._timeout):
                self._state = self.STATE_STOPPING
                self._command_pub.publish(self._disable_msg)
                self._return = errno.ETIMEDOUT

            elif self._state in (self.STATE_STARTING, self.STATE_RUNNING):
                self._command_pub.publish(self._enable_msg)

            elif self._state == self.STATE_STOPPING:
                self._command_pub.publish(self._disable_msg)

            elif self._state == self.STATE_IDLE:
                break

            time.sleep(0.5) # Equivalent to ROS 1 Rate(2) outside of callback context

    def _on_shutdown(self):
        import time
        while not self._state == self.STATE_IDLE:
            self._command_pub.publish(self._disable_msg)
            time.sleep(0.5)

        self._return = errno.ECONNABORTED

    def run(self):
        self._state = self.STATE_STARTING
        self._command_pub.publish(self._enable_msg)
        self._run_loop()
        if self._return != 0:
            msgs = {
                errno.EIO:          "Robust controller failed",
                errno.ENOMSG:       "Robust controller failed to enable",
                errno.ETIMEDOUT:    "Robust controller timed out",
                errno.ECONNABORTED: "Robust controller interrupted by user",
                }

            msg = msgs.get(self._return, None)
            if msg:
                raise IOError(self._return, msg)
            raise IOError(self._return)