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
import roslibpy

class BaxterRoslibIK:
    def __init__(self, limb, host='130.251.13.31', port=9090):
        self.limb = limb
        
        # 1. Connect directly to Baxter's internal ROS 1 websocket
        self.client = roslibpy.Ros(host=host, port=port)
        print(f"Connecting to Baxter rosbridge at ws://{host}:{port}...")
        self.client.run()

        if not self.client.is_connected:
            print("Error: Failed to connect to Baxter. Check network ping.")
            sys.exit(1)
        print("Connected successfully!")

        # 2. Define the IK Service
        ik_ns = f'/ExternalTools/{limb}/PositionKinematicsNode/IKService'
        self.ik_service = roslibpy.Service(self.client, ik_ns, 'baxter_core_msgs/SolvePositionIK')

        # 3. Define the Joint Command Publisher
        pub_ns = f'/robot/limb/{limb}/joint_command'
        self.joint_pub = roslibpy.Topic(self.client, pub_ns, 'baxter_core_msgs/JointCommand')

    def execute_test(self):
        print("Preparing absolute Cartesian coordinates...")

        # Standard Baxter test poses relative to the 'base' frame (the robot's chest)
        if self.limb == 'left':
            pos = {'x': 0.6575, 'y': 0.8519, 'z': 0.0388}
            ori = {'x': -0.3668, 'y': 0.8859, 'z': 0.1081, 'w': 0.2621}
        else:
            pos = {'x': 0.6569, 'y': -0.8525, 'z': 0.0388}
            ori = {'x': 0.3670, 'y': 0.8859, 'z': -0.1089, 'w': 0.2618}

        # Build the JSON request dictionary
        request = roslibpy.ServiceRequest({
            'pose_stamp': [{
                'header': {'frame_id': 'base'},
                'pose': {
                    'position': pos,
                    'orientation': ori
                }
            }],
            'seed_mode': 1  # 1 = SEED_CURRENT (Start calculating from current arm position)
        })

        print(f"Calling IK Service for the {self.limb} arm...")
        
        # Call the service synchronously
        try:
            response = self.ik_service.call(request)
        except Exception as e:
            print(f"Service call failed: {e}")
            return

        # Check if the math solver found a valid configuration
        if response and response.get('isValid', [False])[0]:
            print("\nSUCCESS - Valid Joint Solution Found!")
            
            # Extract the 7 joints from the JSON response
            joint_names = response['joints'][0]['name']
            joint_positions = response['joints'][0]['position']
            
            # Print for terminal verification
            limb_joints = dict(zip(joint_names, joint_positions))
            for joint, angle in limb_joints.items():
                print(f"  {joint}: {angle:.4f}")
            
            print("\nPublishing physical movement command...")
            
            # Build the Joint Command dictionary (Mode 1 is POSITION_MODE)
            cmd_msg = {
                'mode': 1,
                'names': joint_names,
                'command': joint_positions
            }
            
            # Publish multiple times to ensure the UDP/Websocket catches the packet
            for _ in range(5):
                self.joint_pub.publish(roslibpy.Message(cmd_msg))
                time.sleep(0.1)
                
            print("Movement execution completed.")
        else:
            print("INVALID POSE - No Valid Joint Solution Found. Coordinate is out of reach.")

    def close(self):
        self.joint_pub.unadvertise()
        self.client.terminate()

def main():
    parser = argparse.ArgumentParser(description="Baxter WebSocket IK Client")
    parser.add_argument('-l', '--limb', choices=['left', 'right'], required=True)
    
    # parse_known_args prevents ROS 2 internal launch arguments from crashing the script
    args, _ = parser.parse_known_args()

    ik_client = BaxterRoslibIK(args.limb)
    ik_client.execute_test()
    ik_client.close()

if __name__ == '__main__':
    main()