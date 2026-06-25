import roslibpy
import time
import sys
import argparse

def main():
    # 1. Set up terminal argument parsing
    parser = argparse.ArgumentParser(description="Multi-Limb Live Cartesian Coordinate Tracker")
    parser.add_argument(
        '-l', '--limb', 
        choices=['left', 'right'], 
        default='right',
        help="The arm to track (left or right). Defaults to right if omitted."
    )
    args, _ = parser.parse_known_args()

    # 2. Connect to Baxter's internal WebSocket core
    client = roslibpy.Ros(host='130.251.13.31', port=9090)
    try:
        client.run()
    except Exception as exc:
        print(f"Error: could not connect to Baxter at ws://130.251.13.31:9090: {exc}\n"
              "Check that the robot is on, rosbridge_server is running, and the host/port are correct.",
              file=sys.stderr)
        sys.exit(1)
    
    if not client.is_connected:
        print("Error: Cannot reach Baxter. Verify your network connection.")
        sys.exit(1)

    print(f"\n>>> Connected! Zero-Lag Callback Stream Active for {args.limb.upper()} arm. <<<")
    print(f"Move the {args.limb} arm by the cuff. Coordinates will update live down the screen.")
    print("Press Ctrl+C to freeze and copy your target coordinates.\n")
    print("-" * 60)

    # Tracking state for our pacing filter
    tracking_state = {'last_print_time': 0.0}

    def callback(msg):
        current_time = time.time()
        
        # Pacing filter: Only print every 0.5 seconds (500ms)
        if current_time - tracking_state['last_print_time'] >= 0.5:
            tracking_state['last_print_time'] = current_time
            
            pos = msg['pose']['position']
            ori = msg['pose']['orientation']
            
            # Print snapshot formatted for copy-pasting
            print(f"\n[ LIVE SNAPSHOT - {args.limb.upper()} ARM ]")
            print(f"START_POS = {{'x': {pos['x']:.4f}, 'y': {pos['y']:.4f}, 'z': {pos['z']:.4f}}}")
            print(f"START_ORI = {{'x': {ori['x']:.4f}, 'y': {ori['y']:.4f}, 'z': {ori['z']:.4f}, 'w': {ori['w']:.4f}}}")
            print("-" * 60)
            
            sys.stdout.flush()

    # 3. Dynamically generate the topic string based on terminal selection
    topic_name = f'/robot/limb/{args.limb}/endpoint_state'
    
    topic = roslibpy.Topic(
        client, 
        topic_name, 
        'baxter_core_msgs/EndpointState',
        queue_length=1
    )
    topic.subscribe(callback)

    try:
        # Keep the primary thread alive while the background thread handles printing
        while client.is_connected:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"\nStream for {args.limb} arm stopped safely.")
        topic.unsubscribe()
        client.terminate()

if __name__ == '__main__':
    main()