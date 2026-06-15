import roslibpy
from copy import deepcopy
from .baxter_dataflow import wait_for

class Limb:
    def __init__(self, ros_client, limb):
        self.ros_client = ros_client
        self.name = limb
        self._joint_angle = {}
        self._cartesian_pose = {}
        
        # Subscribe to states
        roslibpy.Topic(self.ros_client, '/robot/joint_states', 'sensor_msgs/JointState').subscribe(self._on_joint_states)
        roslibpy.Topic(self.ros_client, f'/robot/limb/{limb}/endpoint_state', 'baxter_core_msgs/EndpointState').subscribe(self._on_endpoint_states)
        
        self._pub_joint_cmd = roslibpy.Topic(self.ros_client, f'/robot/limb/{limb}/joint_command', 'baxter_core_msgs/JointCommand')

    def _on_joint_states(self, msg):
        for idx, name in enumerate(msg['name']):
            self._joint_angle[name] = msg['position'][idx]

    def _on_endpoint_states(self, msg):
        self._cartesian_pose = msg['pose']

    def set_joint_positions(self, positions):
        msg = {
            'mode': 1, # POSITION_MODE
            'names': list(positions.keys()),
            'command': list(positions.values())
        }
        self._pub_joint_cmd.publish(roslibpy.Message(msg))