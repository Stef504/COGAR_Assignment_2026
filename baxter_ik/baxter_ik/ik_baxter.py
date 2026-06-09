#!/usr/bin/env python3

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

"""
Baxter ROS 2 Inverse Kinematics & Execution Client
Upgraded for baxter_rosbridge_adapter.
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header

from baxter_core_msgs.srv import SolvePositionIK
from baxter_core_msgs.msg import JointCommand


class IKServiceClient(Node):
    def __init__(self, limb):
        super().__init__(f'rsdk_ik_service_client_{limb}')
        self.limb = limb
        
        # 1. Setup the IK Service Client (To calculate the math)
        ik_ns = f'/ExternalTools/{limb}/PositionKinematicsNode/IKService'
        self.ik_client = self.create_client(SolvePositionIK, ik_ns)
        
        # 2. Setup the Joint Command Publisher (To physically move the arm)
        pub_ns = f'/robot/limb/{limb}/joint_command'
        self.joint_pub = self.create_publisher(JointCommand, pub_ns, 10)

    def ik_test(self):
        # Wait for the IK service to become available on the bridge
        while not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('IK Service not available, waiting again...')

        ikreq = SolvePositionIK.Request()
        hdr = Header(stamp=self.get_clock().now().to_msg(), frame_id='base')
        
        # Hardcoded Cartesian poses for testing (from the original SDK)
        poses = {
            'left': PoseStamped(
                header=hdr,
                pose=Pose(
                    position=Point(x=0.657579481614, y=0.851981417433, z=0.0388352386502),
                    orientation=Quaternion(x=-0.366894936773, y=0.885980397775, z=0.108155782462, w=0.262162481772),
                ),
            ),
            'right': PoseStamped(
                header=hdr,
                pose=Pose(
                    position=Point(x=0.656982770038, y=-0.852598021641, z=0.0388609422173),
                    orientation=Quaternion(x=0.367048116303, y=0.885911751787, z=-0.108908281936, w=0.261868353356),
                ),
            ),
        }

        ikreq.pose_stamp.append(poses[self.limb])
        # Tell the IK solver to use the arm's current position as the starting seed for the math
        ikreq.seed_mode = ikreq.SEED_CURRENT

        self.get_logger().info("Sending Cartesian coordinates to IK Service...")
        
        # Call the service synchronously
        future = self.ik_client.call_async(ikreq)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            resp = future.result()
            
            # Check if the internal math found a valid joint configuration (isValid[0] == True)
            if resp.isValid[0]:
                self.get_logger().info("SUCCESS - Valid Joint Solution Found!")
                
                # Format solution into a dictionary for terminal verification
                limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
                print("\nIK Joint Solution:\n", limb_joints)
                print("------------------")
                
                # --- THE EXECUTION PHASE ---
                # Build the actual movement message
                cmd = JointCommand()
                cmd.mode = JointCommand.POSITION_MODE
                cmd.names = resp.joints[0].name
                cmd.command = resp.joints[0].position
                
                self.get_logger().info(f"Publishing command to move the {self.limb} arm...")
                
                # Publish the command to the bridge. 
                # (Looping a few times ensures the ROS 1 bridge catches the UDP packet over the network)
                for _ in range(5):
                    self.joint_pub.publish(cmd)
                    time.sleep(0.1)
                    
                self.get_logger().info("Movement execution completed.")
                return 0
            else:
                self.get_logger().error("INVALID POSE - No Valid Joint Solution Found. Arm cannot reach this coordinate.")
                return 1
        else:
            self.get_logger().error("Service call to Baxter IK failed. Check your bridge connection.")
            return 1


def main(args=None):
    rclpy.init(args=args)
    
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt, description="Baxter ROS 2 IK Client & Movement Execution")
    parser.add_argument(
        '-l', '--limb', choices=['left', 'right'], required=True,
        help="the limb to test and move"
    )
    
    # Parse arguments (slicing sys.argv to avoid catching internal ROS 2 launch args)
    parsed_args = parser.parse_args(sys.argv[1:3])

    ik_node = IKServiceClient(parsed_args.limb)
    result = ik_node.ik_test()
    
    ik_node.destroy_node()
    rclpy.shutdown()
    sys.exit(result)

if __name__ == '__main__':
    main()