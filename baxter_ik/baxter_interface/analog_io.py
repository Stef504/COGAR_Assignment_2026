import errno
import roslibpy
from .baxter_dataflow import wait_for

class AnalogIO(object):
    def __init__(self, ros_client, component_id):
        self.ros_client = ros_client
        self._id = component_id
        self._component_type = 'analog_io'
        self._is_output = False
        self._state = dict()

        type_ns = f'/robot/{self._component_type}'
        topic_base = f'{type_ns}/{self._id}'

        self._sub_state = roslibpy.Topic(self.ros_client, f'{topic_base}/state', 'baxter_core_msgs/AnalogIOState')
        self._sub_state.subscribe(self._on_io_state)

        wait_for(
            lambda: len(self._state) != 0,
            timeout=2.0,
            timeout_msg=f"Failed to get current analog_io state from {topic_base}"
        )

        if self._is_output:
            self._pub_output = roslibpy.Topic(self.ros_client, f'{type_ns}/command', 'baxter_core_msgs/AnalogOutputCommand')

    def _on_io_state(self, msg):
        self._is_output = not msg.get('isInputOnly', False)
        self._state['value'] = msg.get('value', 0)

    def state(self):
        return self._state['value']

    def is_output(self):
        return self._is_output

    def set_output(self, value, timeout=2.0):
        if not self._is_output:
            raise IOError(errno.EACCES, f"Component is not an output [{self._component_type}: {self._id}]")
            
        cmd = roslibpy.Message({'name': self._id, 'value': value})
        self._pub_output.publish(cmd)

        if timeout != 0:
            wait_for(
                test=lambda: self.state() == value,
                timeout=timeout,
                timeout_msg=f"Failed to command analog io to: {value}",
                body=lambda: self._pub_output.publish(cmd)
            )