# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.

import errno
import rclpy

from baxter_core_msgs.msg import (
    CameraControl,
    CameraSettings,
)
from baxter_core_msgs.srv import (
    CloseCamera,
    ListCameras,
    OpenCamera,
)


class CameraController(object):
    """
    Interface class for controlling camera settings on the Baxter robot.
    """

    MODES = [
             (1280, 800), (960, 600), (640, 400),
             (480, 300), (384, 240), (320, 200),
             ]
    CONTROL_AUTO = -1

    def __init__(self, node, name):
        """
        Constructor.

        @param node: ROS 2 node reference
        @param name: camera identifier.
        """
        self._node = node
        self._id = name

        self._list_svc = self._node.create_client(ListCameras, '/cameras/list')
        self._list_svc.wait_for_service(timeout_sec=10.0)
        
        req = ListCameras.Request()
        res = self._list_svc.call(req)
        if not self._id in res.cameras:
            raise AttributeError(
                ("Cannot locate a service for camera name '{0}'. "
                "Close a different camera first and try again.".format(self._id)))

        self._open_svc = self._node.create_client(OpenCamera, '/cameras/open')
        self._close_svc = self._node.create_client(CloseCamera, '/cameras/close')

        self._settings = CameraSettings()
        self._settings.width = 320
        self._settings.height = 200
        self._settings.fps = 20
        self._open = False

    def _reload(self):
        self.open()

    def _get_value(self, control, default):
        lookup = [c.value for c in self._settings.controls if c.id == control]
        try:
            return lookup[0]
        except IndexError:
            return default

    def _set_control_value(self, control, value):
        lookup = [c for c in self._settings.controls if c.id == control]
        try:
            lookup[0].value = value
        except IndexError:
            self._settings.controls.append(CameraControl(id=control, value=value))

    @property
    def resolution(self):
        return (self._settings.width, self._settings.height)

    @resolution.setter
    def resolution(self, res):
        res = tuple(res)
        if len(res) != 2:
            raise AttributeError("Invalid resolution specified")
        if not res in self.MODES:
            raise ValueError("Invalid camera mode %dx%d" % (res[0], res[1]))
        self._settings.width = res[0]
        self._settings.height = res[1]
        self._reload()

    @property
    def fps(self):
        return self._settings.fps

    @fps.setter
    def fps(self, fps):
        self._settings.fps = fps
        self._reload()

    @property
    def exposure(self):
        return self._get_value(CameraControl.CAMERA_CONTROL_EXPOSURE, self.CONTROL_AUTO)

    @exposure.setter
    def exposure(self, exposure):
        exposure = int(exposure)
        if (exposure < 0 or exposure > 100) and exposure != self.CONTROL_AUTO:
            raise ValueError("Invalid exposure value")
        self._set_control_value(CameraControl.CAMERA_CONTROL_EXPOSURE, exposure)
        self._reload()

    @property
    def gain(self):
        return self._get_value(CameraControl.CAMERA_CONTROL_GAIN, self.CONTROL_AUTO)

    @gain.setter
    def gain(self, gain):
        gain = int(gain)
        if (gain < 0 or gain > 79) and gain != self.CONTROL_AUTO:
            raise ValueError("Invalid gain value")
        self._set_control_value(CameraControl.CAMERA_CONTROL_GAIN, gain)
        self._reload()

    @property
    def white_balance_red(self):
        return self._get_value(CameraControl.CAMERA_CONTROL_WHITE_BALANCE_R, self.CONTROL_AUTO)

    @white_balance_red.setter
    def white_balance_red(self, value):
        value = int(value)
        if (value < 0 or value > 4095) and value != self.CONTROL_AUTO:
            raise ValueError("Invalid white balance value")
        self._set_control_value(CameraControl.CAMERA_CONTROL_WHITE_BALANCE_R, value)
        self._reload()

    @property
    def white_balance_green(self):
        return self._get_value(CameraControl.CAMERA_CONTROL_WHITE_BALANCE_G, self.CONTROL_AUTO)

    @white_balance_green.setter
    def white_balance_green(self, value):
        value = int(value)
        if (value < 0 or value > 4095) and value != self.CONTROL_AUTO:
            raise ValueError("Invalid white balance value")
        self._set_control_value(CameraControl.CAMERA_CONTROL_WHITE_BALANCE_G, value)
        self._reload()

    @property
    def white_balance_blue(self):
        return self._get_value(CameraControl.CAMERA_CONTROL_WHITE_BALANCE_B, self.CONTROL_AUTO)

    @white_balance_blue.setter
    def white_balance_blue(self, value):
        value = int(value)
        if (value < 0 or value > 4095) and value != self.CONTROL_AUTO:
            raise ValueError("Invalid white balance value")
        self._set_control_value(CameraControl.CAMERA_CONTROL_WHITE_BALANCE_B, value)
        self._reload()

    @property
    def window(self):
        x = self._get_value(CameraControl.CAMERA_CONTROL_WINDOW_X, self.CONTROL_AUTO)
        if (x == self.CONTROL_AUTO):
            return (tuple(map(lambda x: x / 2, self.resolution)) if self.half_resolution else self.resolution)
        else:
            return (x, self._get_value(CameraControl.CAMERA_CONTROL_WINDOW_Y, self.CONTROL_AUTO))

    @window.setter
    def window(self, win):
        x, y = tuple(win)
        cur_x, cur_y = self.resolution
        limit_x = 1280 - cur_x
        limit_y = 800 - cur_y

        if self.half_resolution:
            limit_x /= 2
            limit_y /= 2

        if x < 0 or x > limit_x:
            raise ValueError("Max X window is %d" % (limit_x,))
        if y < 0 or y > limit_y:
            raise ValueError("Max Y window is %d" % (limit_y,))

        self._set_control_value(CameraControl.CAMERA_CONTROL_WINDOW_X, x)
        self._set_control_value(CameraControl.CAMERA_CONTROL_WINDOW_Y, y)
        self._reload()

    @property
    def flip(self):
        return self._get_value(CameraControl.CAMERA_CONTROL_FLIP, False)

    @flip.setter
    def flip(self, value):
        self._set_control_value(CameraControl.CAMERA_CONTROL_FLIP, int(value != 0))
        self._reload()

    @property
    def mirror(self):
        return self._get_value(CameraControl.CAMERA_CONTROL_MIRROR, False)

    @mirror.setter
    def mirror(self, value):
        self._set_control_value(CameraControl.CAMERA_CONTROL_MIRROR, int(value != 0))
        self._reload()

    @property
    def half_resolution(self):
        return self._get_value(CameraControl.CAMERA_CONTROL_RESOLUTION_HALF, False)

    @half_resolution.setter
    def half_resolution(self, value):
        self._set_control_value(CameraControl.CAMERA_CONTROL_RESOLUTION_HALF, int(value != 0))
        self._reload()

    def open(self):
        if self._id == 'head_camera':
            self._set_control_value(CameraControl.CAMERA_CONTROL_FLIP, True)
            self._set_control_value(CameraControl.CAMERA_CONTROL_MIRROR, True)
        
        req = OpenCamera.Request(name=self._id, settings=self._settings)
        ret = self._open_svc.call(req)
        
        if ret.err != 0:
            raise OSError(ret.err, "Failed to open camera")
        self._open = True

    def close(self):
        req = CloseCamera.Request(name=self._id)
        ret = self._close_svc.call(req)
        if ret.err != 0 and ret.err != errno.EINVAL:
            raise OSError(ret.err, "Failed to close camera")
        self._open = False