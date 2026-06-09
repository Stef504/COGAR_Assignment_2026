import argparse
import sys
import time
import roslibpy

# =========================================================
# 1. PASTE YOUR COORDINATES FROM THE VISUALIZER SCRIPT HERE
# =========================================================
START_POS = {'x': 0.6000, 'y': -0.3000, 'z': 0.1500}
START_ORI = {'x': 0.3670, 'y': 0.8859, 'z': -0.1089, 'w': 0.2618}

# =========================================================
# EXPERIMENT SETTINGS
# =========================================================
SWIPE_DISTANCE_METERS = 0.10  # 10cm forward swipe (adds to X)
REPETITIONS = 10              # Number of forward/backward cycles
DELAY_SECONDS = 2.0           # Delay to allow data separation

class RepetitiveSwiper:
    def __init__(self, limb, host='130.251.13.31', port=9090):
        self.limb = limb
        self.client = roslibpy.Ros(host=host, port=port)
        self.client.run()

        if not self.client.is_connected:
            print("Error: Failed to connect to Baxter.")
            sys.exit(1)

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
            # seed_mode deliberately left out so the solver calculates from current state
        })

        response = self.ik_service.call(request)

        if response and response.get('isValid', [False])[0]:
            cmd_msg = {
                'mode': 1, # POSITION_MODE
                'names': response['joints'][0]['name'],
                'command': response['joints'][0]['position']
            }
            
            # Publish multiple times to ensure network delivery
            for _ in range(5):
                self.joint_pub.publish(roslibpy.Message(cmd_msg))
                time.sleep(0.1)
                
            time.sleep(1.5) # Wait for physical motion to complete
            return True
        else:
            print(f"ERROR: Coordinate X:{x:.3f}, Y:{y:.3f}, Z:{z:.3f} is out of reach!")
            return False

    def run_experiment(self):
        print("\nStarting repetitive experiment...")
        
        base_x = START_POS['x']
        end_x = base_x + SWIPE_DISTANCE_METERS

        for i in range(1, REPETITIONS + 1):
            print(f"\n--- Repetition {i}/{REPETITIONS} ---")
            
            print(f"  -> Moving to Start Position (X: {base_x:.3f})")
            success = self.execute_ik_movement(base_x, START_POS['y'], START_POS['z'], START_ORI)
            if not success: break
            time.sleep(DELAY_SECONDS)

            print(f"  -> Swiping Forward to End Position (X: {end_x:.3f})")
            success = self.execute_ik_movement(end_x, START_POS['y'], START_POS['z'], START_ORI)
            if not success: break
            time.sleep(DELAY_SECONDS)

        print("\nExperiment loops finished.")

    def close(self):
        self.joint_pub.unadvertise()
        self.client.terminate()

def main():
    parser = argparse.ArgumentParser(description="Repetitive IK Executer")
    parser.add_argument('-l', '--limb', choices=['left', 'right'], default='right')
    args, _ = parser.parse_known_args()

    swiper = RepetitiveSwiper(args.limb)
    try:
        swiper.run_experiment()
    except KeyboardInterrupt:
        print("\nExperiment interrupted.")
    finally:
        swiper.close()

if __name__ == '__main__':
    main()