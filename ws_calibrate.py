import time
import roslibpy

def main():
    print("========================================")
    print("  BAXTER WEBSOCKET CALIBRATION TOOL")
    print("========================================")
    
    limb = input("Which arm are you calibrating? (left/right): ").strip().lower()
    if limb not in ['left', 'right']:
        print("Invalid arm. Exiting.")
        return

    host = '130.251.13.31'
    port = 9090

    print(f"\n[1] Connecting to Baxter at ws://{host}:{port}...")
    client = roslibpy.Ros(host=host, port=port)
    client.run()

    if not client.is_connected:
        print("CRITICAL ERROR: Failed to connect to Baxter.")
        return
    print("Connected successfully!")

    # --- THE MISSING LINK: HARDWARE RESET ---
    print("\n[2] Clearing internal safety locks (Hardware Reset)...")
    reset_pub = roslibpy.Topic(client, '/robot/set_super_reset', 'std_msgs/Empty')
    reset_pub.publish(roslibpy.Message({}))
    time.sleep(2.0) # Give the motherboard time to clear the errors

    # 3. Enable the Robot
    print("\n[3] Enabling Robot Motors...")
    print("    >>> LISTEN CLOSELY: You should hear a physical 'clunk' sound.")
    enable_pub = roslibpy.Topic(client, '/robot/set_super_enable', 'std_msgs/Bool')
    enable_pub.publish(roslibpy.Message({'data': True}))
    time.sleep(2.0) # Give motors time to engage

    # 4. Start Calibration
    topic_name = f'/robustcontroller/{limb}/CalibrateArm/enable'
    calib_pub = roslibpy.Topic(client, topic_name, 'baxter_maintenance_msgs/CalibrateArmEnable')

    print(f"\n[4] STARTING {limb.upper()} ARM CALIBRATION!")
    print(">>> WARNING: DO NOT TOUCH THE ARM. <<<")
    print(">>> Wait for the arm to stop moving completely (approx 2-5 minutes).")
    print(">>> Press Ctrl+C ONLY when the arm has returned to a resting state.\n")

    try:
        # The 10Hz Heartbeat Loop
        ticks = 0
        while client.is_connected:
            calib_pub.publish(roslibpy.Message({
                'isEnabled': True, 
                'uid': 'daimon_websocket_calib'
            }))
            time.sleep(0.1) 
            
            ticks += 1
            if ticks % 100 == 0:
                print(f"Calibration in progress... ({ticks / 10:.0f} seconds elapsed)")
                
    except KeyboardInterrupt:
        print("\n\n[!] Manual stop triggered.")
        
    finally:
        print("Shutting down calibration controllers...")
        calib_pub.publish(roslibpy.Message({
            'isEnabled': False, 
            'uid': 'daimon_websocket_calib'
        }))
        calib_pub.unadvertise()
        enable_pub.unadvertise()
        reset_pub.unadvertise()
        client.terminate()
        print("Calibration routine finished safely.")

if __name__ == '__main__':
    main()