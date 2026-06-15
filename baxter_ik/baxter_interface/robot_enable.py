import roslibpy
from .baxter_dataflow import wait_for

class RobotEnable:
    def __init__(self, ros_client):
        self.ros_client = ros_client
        self._state = None
        self._state_sub = roslibpy.Topic(self.ros_client, '/robot/state', 'baxter_core_msgs/AssemblyState')
        self._state_sub.subscribe(lambda msg: setattr(self, '_state', msg))
        wait_for(lambda: self._state is not None, timeout=2.0)

    def _toggle_enabled(self, status):
        pub = roslibpy.Topic(self.ros_client, '/robot/set_super_enable', 'std_msgs/Bool')
        pub.publish(roslibpy.Message({'data': status}))
        wait_for(lambda: self._state.get('enabled') == status, timeout=5.0)
        pub.unadvertise()

    def enable(self): self._toggle_enabled(True)
    def disable(self): self._toggle_enabled(False)