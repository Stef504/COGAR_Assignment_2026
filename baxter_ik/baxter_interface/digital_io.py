import errno
import roslibpy
from .baxter_dataflow import wait_for
from .signals import Signal

class DigitalIO(object):
    def __init__(self, ros_client, component_id):
        self.ros_client = ros_client
        self._id = component_id
        self._component_type = 'digital_io'
        self._is_output = False
        self._state = None
        self.state_changed = Signal()

        type_ns = f'/robot/{self._component_type}'
        topic_base = f'{type_ns}/{self._id}'

        self._sub_state = roslibpy.Topic(self.ros_client, f'{topic_base}/state', 'baxter_core_msgs/DigitalIOState')
        self._sub_state.subscribe(self._on_io_state)

        wait_for(
            lambda: self._state is not None,
            timeout=2.0,
            timeout_msg=f"Failed to get current digital_io state from {topic_base}"
        )

        if self._is_output:
            self._pub_output = roslibpy.Topic(self.ros_client, f'{type_ns}/command', 'baxter_core_msgs/DigitalOutputCommand')

    def _on_io_state(self, msg):
        new_state = (msg.get('state') == 1) # 1 represents PRESSED
        if self._state is None:
            self._is_output = not msg.get('isInputOnly', False)
        
        old_state = self._state
        self._state = new_state

        if old_state is not None and old_state != new_state:
            self.state_changed(new_state)

    @property
    def is_output(self):
        return self._is_output

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self.set_output(value)

    def set_output(self, value, timeout=2.0):
        if not self._is_output:
            raise IOError(errno.EACCES, f"Component is not an output [{self._component_type}: {self._id}]")
            
        cmd = roslibpy.Message({'name': self._id, 'value': value})
        self._pub_output.publish(cmd)

        if timeout != 0:
            wait_for(
                test=lambda: self.state == value,
                timeout=timeout,
                timeout_msg=f"Failed to command digital io to: {value}",
                body=lambda: self._pub_output.publish(cmd)
            )