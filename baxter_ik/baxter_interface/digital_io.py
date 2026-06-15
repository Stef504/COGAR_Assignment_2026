class WebSocketDigitalIO:
    def __init__(self, client, component_id):
        self.client = client
        self._id = component_id
        self._state = None
        self._is_output = False
        
        topic_base = f'/robot/digital_io/{self._id}'
        
        self._sub_state = roslibpy.Topic(self.client, f'{topic_base}/state', 'baxter_core_msgs/DigitalIOState')
        self._sub_state.subscribe(self._on_io_state)
        
        self._pub_output = roslibpy.Topic(self.client, f'{topic_base}/command', 'baxter_core_msgs/DigitalOutputCommand')

    def _on_io_state(self, msg):
        self._state = (msg['state'] == 1) # 1 is PRESSED
        self._is_output = not msg.get('is_input_only', True)

    def set_output(self, value):
        """Sets the LED or digital output state (True/False)."""
        if self._is_output:
            msg = {'name': self._id, 'value': bool(value)}
            self._pub_output.publish(roslibpy.Message(msg))