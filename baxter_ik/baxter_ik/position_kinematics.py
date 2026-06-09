import roslibpy
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Live Cartesian Coordinate Visualizer")
    parser.add_argument('-l', '--limb', choices=['left', 'right'], default='right')
    args, _ = parser.parse_known_args()

    # Connect to the robot's websocket
    client = roslibpy.Ros(host='130.251.13.31', port=9090)
    print(f"Connecting to Baxter at ws://130.251.13.31:9090...")
    client.run()
    
    if not client.is_connected:
        print("Failed to connect.")
        sys.exit(1)

    print(f"Connected! Tracking the {args.limb} arm.")
    print("Move the arm manually. Press Ctrl+C to stop and copy the values.\n")

    def callback(msg):
        pos = msg['pose']['position']
        ori = msg['pose']['orientation']
        # Clear the terminal screen slightly for readability
        sys.stdout.write('\033[2J\033[H') 
        print("=========================================")
        print(f"   LIVE COORDINATES: {args.limb.upper()} ARM")
        print("=========================================")
        print("PASTE THIS BLOCK INTO YOUR IK SCRIPT:")
        print(f"pos = {{'x': {pos['x']:.4f}, 'y': {pos['y']:.4f}, 'z': {pos['z']:.4f}}}")
        print(f"ori = {{'x': {ori['x']:.4f}, 'y': {ori['y']:.4f}, 'z': {ori['z']:.4f}, 'w': {ori['w']:.4f}}}")
        print("=========================================")
        sys.stdout.flush()
        
    # Subscribe to the endpoint state topic
    topic = roslibpy.Topic(client, f'/robot/limb/{args.limb}/endpoint_state', 'baxter_core_msgs/EndpointState',throttle_rate=500)
    topic.subscribe(callback)

    try:
        while client.is_connected:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDisconnecting visualizer...")
        topic.unsubscribe()
        client.terminate()

if __name__ == '__main__':
    main()