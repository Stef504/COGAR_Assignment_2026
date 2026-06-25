import roslibpy
import time
import sys

def main():
    # Connect to Baxter
    client = roslibpy.Ros(host='130.251.13.31', port=9090)
    try:
        client.run()
    except Exception as exc:
        print(f"Error: cannot reach Baxter at ws://130.251.13.31:9090: {exc}\n"
              "Check that the robot is on, rosbridge_server is running, and the host/port are correct.",
              file=sys.stderr)
        sys.exit(1)
    
    if not client.is_connected:
        print("CRITICAL: Cannot connect to Baxter's WebSocket server.")
        sys.exit(1)
        
    print("\nConnected! Listening for ANY data on the endpoint topic...")

    # Global packet counter
    packet_count = 0

    def callback(msg):
        nonlocal packet_count
        packet_count += 1
        print(f"\n[PACKET RECEIVED #{packet_count}]")
        print("Raw Message Keys available in this packet:")
        print(msg.keys()) # Shows if 'pose', 'twist', or 'wrench' exist
        if 'pose' in msg:
            print("Pose internal structure:", msg['pose'].keys())

    # Subscribe without any throttle limitations
    topic = roslibpy.Topic(client, '/robot/limb/right/endpoint_state', 'baxter_core_msgs/EndpointState')
    topic.subscribe(callback)

    try:
        while client.is_connected:
            time.sleep(1.0)
            if packet_count == 0:
                print("Still waiting for packets... Topic is completely silent.")
    except KeyboardInterrupt:
        topic.unsubscribe()
        client.terminate()

if __name__ == '__main__':
    main()