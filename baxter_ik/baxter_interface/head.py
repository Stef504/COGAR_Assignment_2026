class WebSocketHead:
    def __init__(self, client):
        self.client = client
        self._state = {}
        
        self._pub_pan = roslibpy.Topic(self.client, '/robot/head/command_head_pan', 'baxter_core_msgs/HeadPanCommand')
        self._pub_nod = roslibpy.Topic(self.client, '/robot/head/command_head_nod', 'std_msgs/Bool')
        
        self._sub_state = roslibpy.Topic(self.client, '/robot/head/head_state', 'baxter_core_msgs/HeadState')
        self._sub_state.subscribe(self._on_head_state)

    def _on_head_state(self, msg):
        self._state = msg

    def set_pan(self, angle, speed_ratio=1.0):
        """Pans the head to a specific angle (radians)."""
        msg = {
            'target': float(angle),
            'speed_ratio': float(speed_ratio),
            'enable_pan_request': True
        }
        self._pub_pan.publish(roslibpy.Message(msg))

    def command_nod(self):
        """Commands Baxter to nod its head."""
        self._pub_nod.publish(roslibpy.Message({'data': True}))
        time.sleep(1.0)