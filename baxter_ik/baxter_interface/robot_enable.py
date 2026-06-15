class WebSocketRobotEnable:
    def __init__(self, client):
        self.client = client
        self._state = {}
        self._state_sub = roslibpy.Topic(self.client, '/robot/state', 'baxter_core_msgs/AssemblyState')
        self._state_sub.subscribe(self._state_callback)

        self._pub_enable = roslibpy.Topic(self.client, '/robot/set_super_enable', 'std_msgs/Bool')
        self._pub_reset = roslibpy.Topic(self.client, '/robot/set_super_reset', 'std_msgs/Empty')
        self._pub_stop = roslibpy.Topic(self.client, '/robot/set_super_stop', 'std_msgs/Empty')

        print("Waiting for Baxter's motherboard state...")
        timeout = time.time() + 5.0
        while not self._state and self.client.is_connected:
            if time.time() > timeout:
                print("[ERROR] Timeout waiting for robot state.")
                break
            time.sleep(0.1)

    def _state_callback(self, msg):
        self._state = msg # msg is already a parsed JSON dictionary

    def state(self):
        return self._state

    def enable(self):
        if self._state.get('stopped', False):
            print("Robot Stopped: Attempting Reset...")
            self.reset()
            
        print("Enabling Robot Motors...")
        self._pub_enable.publish(roslibpy.Message({'data': True}))
        time.sleep(2.0)

    def disable(self):
        print("Disabling Robot Motors...")
        self._pub_enable.publish(roslibpy.Message({'data': False}))

    def reset(self):
        if self._state.get('stopped', False) and self._state.get('estop_button', 0) == 1:
            print("[FATAL] E-Stop is physically ASSERTED. Cannot reset.")
            return False

        print("Resetting robot safety locks...")
        self._pub_reset.publish(roslibpy.Message({}))
        time.sleep(2.0)
        return True

    def stop(self):
        print("Triggering software STOP...")
        self._pub_stop.publish(roslibpy.Message({}))