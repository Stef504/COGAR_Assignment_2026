class WebSocketGripper:
    def __init__(self, client, gripper_name):
        self.client = client
        self.name = f"{gripper_name}_gripper"
        self._state = {}
        self._prop = {}
        self._cmd_sequence = 0
        
        ns = f'/robot/end_effector/{self.name}/'
        
        # Publishers
        self._cmd_pub = roslibpy.Topic(self.client, ns + 'command', 'baxter_core_msgs/EndEffectorCommand')
        
        # Subscribers
        self._state_sub = roslibpy.Topic(self.client, ns + 'state', 'baxter_core_msgs/EndEffectorState')
        self._state_sub.subscribe(self._on_gripper_state)
        
        self._prop_sub = roslibpy.Topic(self.client, ns + 'properties', 'baxter_core_msgs/EndEffectorProperties')
        self._prop_sub.subscribe(self._on_gripper_prop)

        print(f"Waiting for {self.name} state data...")
        timeout = time.time() + 5.0
        while (not self._state or not self._prop) and self.client.is_connected:
            if time.time() > timeout:
                print(f"[WARNING] Timeout waiting for {self.name}. (Is it plugged in?)")
                break
            time.sleep(0.1)

    def _on_gripper_state(self, msg):
        self._state = msg

    def _on_gripper_prop(self, msg):
        self._prop = msg

    def hardware_id(self):
        return self._state.get('id', -1)

    def command(self, cmd_string, args=None):
        """Sends a raw command to the gripper."""
        self._cmd_sequence = (self._cmd_sequence % 0x7FFFFFFF) + 1
        
        args_json = ""
        if args is not None:
            args_json = json.dumps(args) # EndEffectorCommand requires args to be a string
            
        msg = {
            'id': self.hardware_id(),
            'command': cmd_string,
            'sender': f"websocket_{cmd_string}",
            'sequence': self._cmd_sequence,
            'args': args_json
        }
        self._cmd_pub.publish(roslibpy.Message(msg))

    def open(self):
        self.command('go', args={'position': 100.0})
        time.sleep(0.5)

    def close(self):
        self.command('go', args={'position': 0.0})
        time.sleep(0.5)
        
    def calibrate(self):
        self.command('calibrate')
        time.sleep(2.0)