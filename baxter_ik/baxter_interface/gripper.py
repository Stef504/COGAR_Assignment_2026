import json
import roslibpy
from .baxter_dataflow import wait_for

class Gripper:
    def __init__(self, ros_client, gripper):
        self.ros_client = ros_client
        self.name = gripper + '_gripper'
        self._state = None
        self._cmd_pub = roslibpy.Topic(self.ros_client, f'robot/end_effector/{self.name}/command', 'baxter_core_msgs/EndEffectorCommand')
        roslibpy.Topic(self.ros_client, f'robot/end_effector/{self.name}/state', 'baxter_core_msgs/EndEffectorState').subscribe(lambda m: setattr(self, '_state', m))
        
        wait_for(lambda: self._state is not None, timeout=5.0)

    def command(self, cmd, args=None):
        msg = {
            'id': self._state.get('id', 4294967295),
            'command': cmd,
            'args': json.dumps(args) if args else ''
        }
        self._cmd_pub.publish(roslibpy.Message(msg))

    def open(self): self.command('go', {"position": 100.0})
    def close(self): self.command('go', {"position": 0.0})