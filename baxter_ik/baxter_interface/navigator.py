import roslibpy
from .baxter_dataflow import wait_for
from .signals import Signal
from .digital_io import DigitalIO

class Navigator(object):
    __LOCATIONS = ('left', 'right', 'torso_left', 'torso_right')

    def __init__(self, ros_client, location):
        if location not in self.__LOCATIONS:
            raise AttributeError(f"Invalid Navigator name '{location}'")
            
        self.ros_client = ros_client
        self._id = location
        self._state = None
        self.button0_changed = Signal()
        self.button1_changed = Signal()
        self.button2_changed = Signal()
        self.wheel_changed = Signal()

        nav_state_topic = f'/robot/navigators/{self._id}_navigator/state'
        self._state_sub = roslibpy.Topic(self.ros_client, nav_state_topic, 'baxter_core_msgs/NavigatorState')
        self._state_sub.subscribe(self._on_state)

        self._inner_led = DigitalIO(self.ros_client, f'{self._id}_inner_light')
        self._inner_led_idx = 0

        self._outer_led = DigitalIO(self.ros_client, f'{self._id}_outer_light')
        self._outer_led_idx = 1

        wait_for(lambda: self._state is not None, timeout_msg=f"Navigator init failed from {nav_state_topic}")

    @property
    def wheel(self): return self._state.get('wheel', 0)
    @property
    def button0(self): return self._state.get('buttons', [0])[0]
    @property
    def inner_led(self): return self._state.get('lights', [False])[self._inner_led_idx]
    
    @inner_led.setter
    def inner_led(self, enable): self._inner_led.set_output(enable)

    def _on_state(self, msg):
        if not self._state:
            self._state = msg
            try: self._inner_led_idx = msg.get('light_names', []).index("inner")
            except ValueError: pass
            try: self._outer_led_idx = msg.get('light_names', []).index("outer")
            except ValueError: pass

        if self._state == msg:
            return

        old_state = self._state
        self._state = msg

        buttons = [self.button0_changed, self.button1_changed, self.button2_changed]
        for i, signal in enumerate(buttons):
            if old_state['buttons'][i] != msg['buttons'][i]:
                signal(msg['buttons'][i])

        if old_state['wheel'] != msg['wheel']:
            diff = msg['wheel'] - old_state['wheel']
            if abs(diff % 256) < 127:
                self.wheel_changed(diff % 256)
            else:
                self.wheel_changed(diff % (-256))