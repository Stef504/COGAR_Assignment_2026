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

Baxter ROS 2 Inverse Kinematics & Execution Client
Upgraded with Slower Trajectory Interpolation.
"""

import sys
import time
import roslibpy

class InteractiveSwiper:
    def __init__(self, limb, host='130.251.13.31', port=9090):
        self.limb = limb
        self.client = roslibpy.Ros(host=host, port=port)
        print(f"\nConnecting to Baxter at ws://{host}:{port}...")
        self.client.run()

        if not self.client.is_connected:
            print("Error: Failed to connect to Baxter's core network.")
            sys.exit(1)
        print("Connected successfully!")

        self.status_pub = roslibpy.Topic(self.client, '/tactile_experiment/status', 'std_msgs/String')

        ik_ns = f'/ExternalTools/{limb}/PositionKinematicsNode/IKService'
        self.ik_service = roslibpy.Service(self.client, ik_ns, 'baxter_core_msgs/SolvePositionIK')
        
        pub_ns = f'/robot/limb/{limb}/joint_command'
        self.joint_pub = roslibpy.Topic(self.client, pub_ns, 'baxter_core_msgs/JointCommand')

    def get_ik_solution(self, x, y, z, orientation):
        """Only calculates the IK mathematically. Returns joint angles WITHOUT moving the arm."""
        request = roslibpy.ServiceRequest({
            'pose_stamp': [{
                'header': {'frame_id': 'base'},
                'pose': {
                    'position': {'x': x, 'y': y, 'z': z},
                    'orientation': orientation
                }
            }]
        })
        response = self.ik_service.call(request)
        if response and response.get('isValid', [False])[0]:
            return response['joints'][0]['name'], response['joints'][0]['position']
        return None, None

    def execute_ik_movement(self, x, y, z, orientation):
        """Moves the arm to a location quickly (used for resetting to start position)."""
        names, positions = self.get_ik_solution(x, y, z, orientation)
        if names and positions:
            cmd_msg = {'mode': 1, 'names': names, 'command': positions}
            for _ in range(5):
                self.joint_pub.publish(roslibpy.Message(cmd_msg))
                time.sleep(0.1)
            time.sleep(1.5) 
            return True
        else:
            print(f"  [IK ERROR] Target position is mathematically unreachable!")
            return False

    def execute_slow_trajectory(self, start_joints, end_joints, duration_sec):
        """Slices the movement into tiny waypoints and forces a slow, constrained movement."""
        rate_hz = 20  # 20 updates per second for smooth motion
        steps = int(duration_sec * rate_hz)
        if steps == 0: steps = 1
        sleep_time = duration_sec / steps

        names = start_joints[0]
        start_pos = start_joints[1]
        end_pos = end_joints[1]

        for step in range(steps + 1):
            t = step / float(steps) # Percentage of completion (0.0 to 1.0)
            
            # Linear interpolation for all 7 joints
            current_pos = [s + t * (e - s) for s, e in zip(start_pos, end_pos)]

            cmd_msg = {
                'mode': 1, 
                'names': names,
                'command': current_pos
            }
            self.joint_pub.publish(roslibpy.Message(cmd_msg))
            time.sleep(sleep_time)

        time.sleep(1.0) # Settle at the end point

    def run_loop(self, start_pos, start_ori, reps, distance, orientation, delay, swipe_duration):
        """Executes the automatic swiping repetitions."""
        base_y = start_pos['y']
        end_y = base_y + distance

        print("\n[Pre-Calculating Kinematics...]")
        start_joints = self.get_ik_solution(start_pos['x'], base_y, start_pos['z'], start_ori)
        end_joints = self.get_ik_solution(start_pos['x'], end_y, start_pos['z'], start_ori)

        if start_joints[0] is None or end_joints[0] is None:
            print("CRITICAL IK ERROR: Start or End position is mathematically unreachable! Aborting.")
            return

        print("\n" + "="*40)
        print("         RUNNING EXPERIMENT")
        print("="*40)
        print(f"Limb: {self.limb.upper()} | Cycles: {reps} | Distance: {distance*100} cm along Y-axis")

        for i in range(1, reps + 1):
            print(f"\n--- Cycle {i}/{reps} ---")
            
            # 1. Reset to the starting position quickly
            print(f"  -> Resetting to Start Position (Y: {base_y:.4f})")
            self.status_pub.publish(roslibpy.Message({'data': f"START_{orientation}_REP_{i}"}))
            self.execute_ik_movement(start_pos['x'], base_y, start_pos['z'], start_ori)
            time.sleep(delay)

            # 2. Swipe slowly across the material using the constrained duration
            print(f"  -> Swiping Slowly to End Position (Y: {end_y:.4f}) over {swipe_duration} seconds")
            self.execute_slow_trajectory(start_joints, end_joints, swipe_duration)
            time.sleep(delay)
            
            self.status_pub.publish(roslibpy.Message({'data': f"STOP"})) 
        
        print("  -> Waiting for Daimon sensor to save local matrix...")
        time.sleep(delay)
        
        self.status_pub.publish(roslibpy.Message({'data': f"EXPERIMENT_COMPLETE"}))
        print("\nExperiment execution complete. Safely closing network...")
        time.sleep(1.0)

    def close(self):
        self.joint_pub.unadvertise()
        self.client.terminate()


def get_float_input(prompt):
    """Helper function to guarantee valid numeric entries from the terminal."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid number. Please enter a valid decimal value.")

def main():
    print("====================================================")
    print("     BAXTER INTERACTIVE EXPERIMENT SETTING TOOL    ")
    print("====================================================")

    limb = ""
    while limb not in ['left', 'right']:
        limb = input("Which arm are you using? (left/right): ").strip().lower()

    print("\n[DATASET ORGANIZATION] -> Choose an orientation label for this experiment run(up,down,NE,SE,SW,NW,left,right).")
    orientation = input("  Enter orientation label: ").strip().lower()
    
    print("\n[ Step 1: Enter Position Coordinates (from live_tracker) ]")
    start_pos = {
        'x': get_float_input("  Enter START X (meters): "),
        'y': get_float_input("  Enter START Y (meters): "),
        'z': get_float_input("  Enter START Z (meters): ")
    }

    print("\n[ Step 2: Enter Orientation Quaternion (from live_tracker) ]")
    start_ori = {
        'x': get_float_input("  Enter Quaternion x: "),
        'y': get_float_input("  Enter Quaternion y: "),
        'z': get_float_input("  Enter Quaternion z: "),
        'w': get_float_input("  Enter Quaternion w: ")
    }

    print("\n[ Step 3: Configure Swipe Parameters ]")
    reps = int(get_float_input("  Enter number of swipe repetitions (e.g., 10): "))
    distance = get_float_input("  Enter linear swipe distance along Y axis (meters, e.g., 0.10): ")
    delay = get_float_input("  Enter stabilization delay between actions (seconds, e.g., 2.0): ")
    
    # --- NEW INPUT FOR SLOW SWIPING ---
    swipe_duration = get_float_input("  Enter exact time the swipe should take (seconds, e.g., 5.0): ")

    swiper = InteractiveSwiper(limb=limb)
    try:
        swiper.run_loop(start_pos, start_ori, reps, distance, orientation, delay, swipe_duration)
    except KeyboardInterrupt:
        print("\nExperiment execution suspended via keyboard break.")
    finally:
        swiper.close()

if __name__ == '__main__':
    main()