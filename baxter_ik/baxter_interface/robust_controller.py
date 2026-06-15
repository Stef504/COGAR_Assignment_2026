class WebSocketRobustController:
    def __init__(self, client, namespace, enable_msg_dict, disable_msg_dict, msg_type):
        """
        Used for running long maintenance routines safely (like Calibration).
        """
        self.client = client
        self._timeout = 600 # 10 minute default timeout
        self.is_running = False
        
        self._command_pub = roslibpy.Topic(self.client, f'{namespace}/enable', msg_type)
        
        self._status_sub = roslibpy.Topic(self.client, f'{namespace}/status', 'baxter_core_msgs/RobustControllerStatus')
        self._status_sub.subscribe(self._callback)
        
        self._enable_msg = enable_msg_dict
        self._disable_msg = disable_msg_dict

    def _callback(self, msg):
        if msg.get('complete', 0) != 0: # 0 means still running
            self.is_running = False

    def run(self):
        """Starts the routine and pumps the 10Hz heartbeat automatically."""
        print("\nStarting Robust Controller Routine...")
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running and self.client.is_connected:
                # Timeout safety
                if (time.time() - start_time) > self._timeout:
                    print("[ERROR] Controller Timed Out!")
                    break
                    
                # Publish the 10Hz heartbeat (equivalent to run_loop)
                self._command_pub.publish(roslibpy.Message(self._enable_msg))
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nRoutine interrupted by user.")
            
        finally:
            print("Safely shutting down controller...")
            self._command_pub.publish(roslibpy.Message(self._disable_msg))
            self._command_pub.unadvertise()
            self._status_sub.unsubscribe()