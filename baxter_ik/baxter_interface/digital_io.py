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
import baxter_dataflow

from baxter_core_msgs.msg import (
    DigitalIOState,
    DigitalOutputCommand,
)


class DigitalIO(object):
    """
    Interface class for a simple Digital Input and/or Output on the
    Baxter robot
    """
    def __init__(self, node, component_id):
        """
        Constructor.

        @param node: ROS 2 node reference
        @param component_id: unique id of the digital component
        """
        self._node = node
        self._id = component_id
        self._component_type = 'digital_io'
        self._is_output = False
        self._state = None

        self.state_changed = baxter_dataflow.Signal()

        type_ns = '/robot/' + self._component_type
        topic_base = type_ns + '/' + self._id

        self._sub_state = self._node.create_subscription(
            DigitalIOState,
            topic_base + '/state',
            self._on_io_state,
            10)

        baxter_dataflow.wait_for(
            lambda: self._state != None,
            timeout=2.0,
            timeout_msg="Failed to get current digital_io state from %s" \
            % (topic_base,),
        )

        if self._is_output:
            self._pub_output = self._node.create_publisher(
                DigitalOutputCommand,
                type_ns + '/command',
                10)

    def _on_io_state(self, msg):
        new_state = (msg.state == DigitalIOState.PRESSED)
        if self._state is None:
            self._is_output = not msg.is_input_only
        old_state = self._state
        self._state = new_state

        if old_state is not None and old_state != new_state:
            self.state_changed(new_state)

    @property
    def is_output(self):
        return self._is_output

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self.set_output(value)

    def set_output(self, value, timeout=2.0):
        if not self._is_output:
            raise IOError(errno.EACCES, "Component is not an output [%s: %s]" %
                (self._component_type, self._id))
        cmd = DigitalOutputCommand()
        cmd.name = self._id
        cmd.value = value
        self._pub_output.publish(cmd)

        if not timeout == 0:
            baxter_dataflow.wait_for(
                test=lambda: self.state == value,
                timeout=timeout,
                rate=100,
                timeout_msg=("Failed to command digital io to: %r" % (value,)),
                body=lambda: self._pub_output.publish(cmd)
            )