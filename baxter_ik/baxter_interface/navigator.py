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

import rclpy
import baxter_dataflow

from baxter_core_msgs.msg import NavigatorState
from baxter_interface import digital_io

class Navigator(object):
    __LOCATIONS = ('left', 'right', 'torso_left', 'torso_right')

    def __init__(self, node, location):
        self._node = node
        if not location in self.__LOCATIONS:
            raise AttributeError("Invalid Navigator name '%s'" % (location,))
        self._id = location
        self._state = None
        self.button0_changed = baxter_dataflow.Signal()
        self.button1_changed = baxter_dataflow.Signal()
        self.button2_changed = baxter_dataflow.Signal()
        self.wheel_changed = baxter_dataflow.Signal()

        nav_state_topic = 'robot/navigators/{0}_navigator/state'.format(self._id)
        self._state_sub = self._node.create_subscription(
            NavigatorState,
            nav_state_topic,
            self._on_state,
            10)

        # Assuming digital_io classes were updated to take the node reference as above
        self._inner_led = digital_io.DigitalIO(self._node, '%s_inner_light' % (self._id,))
        self._inner_led_idx = 0

        self._outer_led = digital_io.DigitalIO(self._node, '%s_outer_light' % (self._id,))
        self._outer_led_idx = 1

        init_err_msg = ("Navigator init failed to get current state from %s" % (nav_state_topic,))
        baxter_dataflow.wait_for(lambda: self._state != None, timeout_msg=init_err_msg)

    # ... Properties and setter logic remains identical.
    
    def _on_state(self, msg):
        if not self._state:
            self._state = msg
            try:
                self._inner_led_idx = self._state.light_names.index("inner")
            except ValueError:
                pass
            try:
                self._outer_led_idx = self._state.light_names.index("outer")
            except ValueError:
                pass
        if self._state == msg:
            return

        old_state = self._state
        self._state = msg

        buttons = [self.button0_changed, self.button1_changed, self.button2_changed]
        for i, signal in enumerate(buttons):
            if old_state.buttons[i] != msg.buttons[i]:
                signal(msg.buttons[i])

        if old_state.wheel != msg.wheel:
            diff = msg.wheel - old_state.wheel
            if abs(diff % 256) < 127:
                self.wheel_changed(diff % 256)
            else:
                self.wheel_changed(diff % (-256))