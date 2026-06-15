import sys
import time
import roslibpy
from websocket_baxter import WebSocketRobotEnable, WebSocketRobustController

def main():
    print("========================================")
    print("  SDK-BASED BAXTER ARM CALIBRATION")
    print("========================================")

    limb = input("Which arm are you calibrating? (left/right): ").strip().lower()
    if limb not in ['left', 'right']:
        print("Invalid arm. Exiting.")
        return

    # Physical safety gate
    print("\n[WARNING] Make sure to remove grippers and other attachments before running calibrate.")
    confirm = input("Are grippers physically removed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Calibration aborted. Please unbolt grippers first.")
        return

    # Network Connection
    host = '130.251.13.31'
    port = 9090

    print(f"\n[1] Connecting to Baxter at ws://{host}:{port}...")
    client = roslibpy.Ros(host=host, port=port)
    client.run()

    if not client.is_connected:
        print("CRITICAL ERROR: Failed to connect to Baxter.")
        sys.exit(1)
    print("Connected successfully!")

    # --- 1. ENABLE ROBOT (Handles state checks & resets automatically) ---
    print("\n[2] Checking Robot State and Enabling...")
    robot = WebSocketRobotEnable(client)
    robot.enable()

    # --- 2. SETUP CALIBRATION CONTROLLER ---
    print(f"\n[3] STARTING {limb.upper()} ARM CALIBRATION!")
    print(">>> WARNING: DO NOT TOUCH THE ARM. <<<")
    print(">>> Wait for the arm to stop moving completely (approx 2-5 minutes).")
    print(">>> Press Ctrl+C ONLY when the arm has returned to a resting state.\n")

    # Define the specific parameters for the CalibrateArm routine
    namespace = f'/robustcontroller/{limb}/CalibrateArm'
    enable_msg = {'isEnabled': True, 'uid': 'daimon_sdk_calib'}
    disable_msg = {'isEnabled': False, 'uid': 'daimon_sdk_calib'}
    msg_type = 'baxter_maintenance_msgs/CalibrateArmEnable'

    # Instantiate the controller using our SDK
    calib_controller = WebSocketRobustController(
        client=client, 
        namespace=namespace, 
        enable_msg_dict=enable_msg, 
        disable_msg_dict=disable_msg, 
        msg_type=msg_type
    )

    # Execute the routine (This handles the 10Hz heartbeat in the background)
    calib_controller.run()

    # --- 3. SAFE SHUTDOWN ---
    print("\n[4] Shutting down...")
    robot.disable()
    client.terminate()
    print("Calibration routine finished safely.")

if __name__ == '__main__':
    main()