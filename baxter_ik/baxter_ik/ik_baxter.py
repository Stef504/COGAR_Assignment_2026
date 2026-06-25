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


import sys
import time
import roslibpy

class InteractiveSwiper:
    def __init__(self, limb, host='130.251.13.31', port=9090):
        self.limb = limb
        self.client = roslibpy.Ros(host=host, port=port)
        print(f"\nConnecting to Baxter at ws://{host}:{port}...")
        try:
            self.client.run()
        except Exception as exc:
            print(f"Error: could not connect to Baxter at ws://{host}:{port}: {exc}\n"
                  "Check that the robot is on, rosbridge_server is running, and the host/port are correct.",
                  file=sys.stderr)
            sys.exit(1)

        if not self.client.is_connected:
            print("Error: Failed to connect to Baxter's core network.", file=sys.stderr)
            sys.exit(1)
        print("Connected successfully!")

        # Broadcasts Baxter's state to the Daimon sensor
        self.status_pub = roslibpy.Topic(self.client, '/tactile_experiment/status', 'std_msgs/String')

        # Setup standard communication services and topics
        ik_ns = f'/ExternalTools/{limb}/PositionKinematicsNode/IKService'
        self.ik_service = roslibpy.Service(self.client, ik_ns, 'baxter_core_msgs/SolvePositionIK')
        
        pub_ns = f'/robot/limb/{limb}/joint_command'
        self.joint_pub = roslibpy.Topic(self.client, pub_ns, 'baxter_core_msgs/JointCommand')

    def execute_ik_movement(self, x, y, z, orientation):
        """Sends coordinates to Baxter's IK solver and physically moves the arm."""
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
            cmd_msg = {
                'mode': 1, # POSITION_MODE
                'names': response['joints'][0]['name'],
                'command': response['joints'][0]['position']
            }
            
            # Publish multiple times to ensure network delivery over the socket
            for _ in range(5):
                self.joint_pub.publish(roslibpy.Message(cmd_msg))
                time.sleep(0.1)
                
            time.sleep(1.5) # Allow physical arm to settle
            return True
        else:
            print(f"  [IK ERROR] Target position (X:{x:.3f}, Y:{y:.3f}, Z:{z:.3f}) is mathematically unreachable!")
            return False

    def run_loop(self, start_pos, start_ori, reps, distance, orientation ,delay):
        """Executes the automatic swiping repetitions."""
        base_y = start_pos['y']
        end_y = base_y + distance

        print("\n" + "="*40)
        print("         RUNNING EXPERIMENT")
        print("="*40)
        print(f"Limb: {self.limb.upper()} | Cycles: {reps} | Distance: {distance*100} cm")

        for i in range(1, reps + 1):
            print(f"\n--- Cycle {i}/{reps} ---")
            
            print(f"  -> Moving to Start Position (Y: {base_y:.4f})")
            self.status_pub.publish(roslibpy.Message({'data': f"START_{orientation}_REP_{i}"}))
            if not self.execute_ik_movement(start_pos['x'],base_y, start_pos['z'], start_ori):
                print("Aborting experiment loop due to IK failure.")
                break
            time.sleep(delay)

            print(f"  -> Swiping Forward to End Position (Y: {end_y:.4f})")
            if not self.execute_ik_movement(start_pos['x'],end_y, start_pos['z'], start_ori):
                print("Aborting experiment loop due to IK failure.")
                break
            time.sleep(delay)
            self.status_pub.publish(roslibpy.Message({'data': f"STOP"})) 
        
        time.sleep(delay)
        self.status_pub.publish(roslibpy.Message({'data': f"EXPERIMENT_COMPLETE"}))
        print("\nExperiment execution complete.")

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

    # 1. Target Arm Configuration
    limb = ""
    while limb not in ['left', 'right']:
        limb = input("Which arm are you using? (left/right): ").strip().lower()

    #Choice of orientation for dataset organization
    print("\n[DATASET ORGANIZATION] -> Choose an orientation label for this experiment run(up,down,NE,SE,SW,NW,left,right).")
    orientation = input("  Enter orientation label: ").strip().lower()
    # 2. Get Cartesian Target Positions
    print("\n[ Step 1: Enter Position Coordinates (from live_tracker) ]")
    start_pos = {
        'x': get_float_input("  Enter START X (meters): "),
        'y': get_float_input("  Enter START Y (meters): "),
        'z': get_float_input("  Enter START Z (meters): ")
    }

    # 3. Get Orientation Quaternion Values
    print("\n[ Step 2: Enter Orientation Quaternion (from live_tracker) ]")
    start_ori = {
        'x': get_float_input("  Enter Quaternion x: "),
        'y': get_float_input("  Enter Quaternion y: "),
        'z': get_float_input("  Enter Quaternion z: "),
        'w': get_float_input("  Enter Quaternion w: ")
    }

    # 4. Get Experimental Control Constraints
    print("\n[ Step 3: Configure Swipe Parameters ]")
    reps = int(get_float_input("  Enter number of swipe repetitions (e.g., 10): "))
    distance = get_float_input("  Enter linear swipe distance along Y axis (meters, e.g., 0.10): ")
    delay = get_float_input("  Enter stabilization delay between actions (seconds, e.g., 2.0): ")

    # Initialize and run the system
    swiper = InteractiveSwiper(limb=limb)
    try:
        swiper.run_loop(start_pos, start_ori, reps, distance, orientation, delay)
    except KeyboardInterrupt:
        print("\nExperiment execution suspended via keyboard break.")
    finally:
        swiper.close()

if __name__ == '__main__':
    main()