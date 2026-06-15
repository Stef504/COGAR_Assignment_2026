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
import time
import rclpy


def wait_for(test, timeout=1.0, raise_on_error=True, rate=100,
             timeout_msg="timeout expired", body=None, node=None):
    """
    Waits until some condition evaluates to true.

    @param test: zero param function to be evaluated
    @param timeout: max amount of time to wait. negative/inf for indefinitely
    @param raise_on_error: raise or just return False
    @param rate: the rate (in Hz) at which to check
    @param timeout_msg: message to supply to the timeout exception
    @param body: optional function to execute while waiting
    @param node: optional ROS 2 Node reference to use ROS clock (defaults to wall time)
    """
    # Use ROS 2 node clock if provided, otherwise use standard system time
    if node:
        start_time = node.get_clock().now().nanoseconds / 1e9
    else:
        start_time = time.time()

    end_time = start_time + timeout
    sleep_duration = 1.0 / rate
    notimeout = (timeout < 0.0) or timeout == float("inf")

    while not test():
        # rclpy.ok() replaces rospy.is_shutdown()
        if not rclpy.ok():
            if raise_on_error:
                raise OSError(errno.ESHUTDOWN, "ROS 2 Shutdown")
            return False

        if node:
            current_time = node.get_clock().now().nanoseconds / 1e9
        else:
            current_time = time.time()

        if (not notimeout) and (current_time >= end_time):
            if raise_on_error:
                raise OSError(errno.ETIMEDOUT, timeout_msg)
            return False
            
        if callable(body):
            body()
            
        # Standard sleep replaces rospy.Rate to avoid needing a node context
        time.sleep(sleep_duration)
        
    return True