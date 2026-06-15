import roslibpy
import time

class RobustController:
    def __init__(self, ros_client, namespace, enable_msg, disable_msg, timeout=60):
        self.ros_client = ros_client
        self._pub = roslibpy.Topic(ros_client, f'{namespace}/enable', 'baxter_maintenance_msgs/CalibrateArmEnable')
        self._enable_msg = roslibpy.Message(enable_msg)
        self._disable_msg = roslibpy.Message(disable_msg)
        self._timeout = timeout
        self._running = False

    def run(self):
        self._running = True
        start = time.time()
        while self._running and (time.time() - start) < self._timeout:
            self._pub.publish(self._enable_msg)
            time.sleep(1.0)
        self._pub.publish(self._disable_msg)