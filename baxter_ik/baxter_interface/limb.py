import time
import roslibpy

class WebSocketLimb:
    def __init__(self, client, limb):
        """
        Translated Limb Interface for WebSockets.
        Pass the active roslibpy.Ros client into this class.
        """
        self.client = client
        self.name = limb
        self._joint_angle = {}
        self._cartesian_pose = {}

        # 1. TRANSLATED PUBLISHER (Replaces rospy.Publisher)
        self._pub_joint_cmd = roslibpy.Topic(
            self.client, 
            f'/robot/limb/{limb}/joint_command', 
            'baxter_core_msgs/JointCommand'
        )

        # 2. TRANSLATED SUBSCRIBERS (Replaces rospy.Subscriber)
        self._sub_endpoint = roslibpy.Topic(
            self.client, 
            f'/robot/limb/{limb}/endpoint_state', 
            'baxter_core_msgs/EndpointState'
        )
        self._sub_endpoint.subscribe(self._on_endpoint_states)

        self._sub_joint_states = roslibpy.Topic(
            self.client, 
            '/robot/joint_states', 
            'sensor_msgs/JointState'
        )
        self._sub_joint_states.subscribe(self._on_joint_states)

        # 3. TRANSLATED DATAFLOW (Replaces baxter_dataflow.wait_for)
        print(f"Waiting for {self.name} limb state data...")
        timeout = time.time() + 5.0
        while not self._cartesian_pose and self.client.is_connected:
            if time.time() > timeout:
                print("Timeout waiting for limb data.")
                break
            time.sleep(0.1)

    def _on_joint_states(self, msg):
        """Callback: Updates joint angles using JSON dictionary syntax."""
        for idx, name in enumerate(msg['name']):
            if self.name in name: 
                self._joint_angle[name] = msg['position'][idx]

    def _on_endpoint_states(self, msg):
        """Callback: Updates cartesian pose using JSON dictionary syntax."""
        self._cartesian_pose = {
            'position': msg['pose']['position'],      # Replaces msg.pose.position
            'orientation': msg['pose']['orientation'] # Replaces msg.pose.orientation
        }

    def endpoint_pose(self):
        """Returns the live Cartesian endpoint pose."""
        return self._cartesian_pose

    def joint_angles(self):
        """Returns the live joint angles."""
        return self._joint_angle

    def set_joint_positions(self, positions):
        """
        Commands the joints to specified positions.
        'positions' should be a dictionary: {'left_s0': 0.5, ...}
        """
        cmd_msg = {
            'mode': 1, # POSITION_MODE
            'names': list(positions.keys()),
            'command': list(positions.values())
        }
        self._pub_joint_cmd.publish(roslibpy.Message(cmd_msg))