#!/usr/bin/python2

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

#!/usr/bin/env python3

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# [License truncated for brevity...]

import argparse
import sys

import rclpy
from rclpy.node import Node

import baxter_interface
from baxter_interface import CHECK_VERSION

from baxter_maintenance_msgs.msg import (
    CalibrateArmEnable,
)


class CalibrateArm(baxter_interface.RobustController):
    def __init__(self, node, limb):
        """
        Wrapper to run the CalibrateArm RobustController.

        @param node: ROS 2 node reference
        @param limb: Limb to run CalibrateArm on [left/right]
        """
        enable_msg = CalibrateArmEnable(is_enabled=True, uid='sdk')
        disable_msg = CalibrateArmEnable(is_enabled=False, uid='sdk')

        # Initialize RobustController, use 10 minute timeout for the
        # CalibrateArm process
        super(CalibrateArm, self).__init__(
            node,
            'robustcontroller/%s/CalibrateArm' % (limb,),
            enable_msg,
            disable_msg,
            10 * 60)


def gripper_removed(node, side):
    """
    Verify grippers are removed for calibration/tare.
    """
    gripper = baxter_interface.Gripper(node, side)
    if gripper.type() != 'custom':
        node.get_logger().error("Cannot calibrate with grippers attached."
                                " Remove grippers before calibration!")
        return False
    return True


def main(args=None):
    if args is None:
        args = sys.argv

    # Initialize rclpy first to handle ROS arguments
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-l', '--limb', required=True,
                        choices=['left', 'right'],
                        help="Calibrate the specified arm")
    
    # Parse arguments, ignoring ROS-specific arguments
    parsed_args, _ = parser.parse_known_args(args[1:])
    arm = parsed_args.limb

    print("Initializing node...")
    node = rclpy.create_node('rsdk_calibrate_arm_%s' % (arm,))

    print("Preparing to calibrate...")
    gripper_warn = ("\nIMPORTANT: Make sure to remove grippers and other"
                    " attachments before running calibrate.\n")
    print(gripper_warn)
    
    if not gripper_removed(node, arm):
        node.destroy_node()
        rclpy.shutdown()
        return 1

    rs = baxter_interface.RobotEnable(node, CHECK_VERSION)
    rs.enable()
    cat = CalibrateArm(node, arm)
    node.get_logger().info("Running calibrate on %s arm" % (arm,))

    error = None
    try:
        cat.run()
    except OSError as e:
        error = e.strerror
    except Exception as e:
        error = str(e)
    finally:
        try:
            rs.disable()
        except Exception:
            pass

    if error is None:
        node.get_logger().info("Calibrate arm finished")
    else:
        node.get_logger().error("Calibrate arm failed: %s" % (error,))

    node.destroy_node()
    rclpy.shutdown()
    return 0 if error is None else 1

if __name__ == '__main__':
    sys.exit(main())