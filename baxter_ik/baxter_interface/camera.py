import errno
import roslibpy

class CameraController(object):
    MODES = [(1280, 800), (960, 600), (640, 400), (480, 300), (384, 240), (320, 200)]
    CONTROL_AUTO = -1

    def __init__(self, ros_client, name):
        self.ros_client = ros_client
        self._id = name

        self._list_svc = roslibpy.Service(self.ros_client, '/cameras/list', 'baxter_core_msgs/ListCameras')
        self._open_svc = roslibpy.Service(self.ros_client, '/cameras/open', 'baxter_core_msgs/OpenCamera')
        self._close_svc = roslibpy.Service(self.ros_client, '/cameras/close', 'baxter_core_msgs/CloseCamera')

        # Synchronous check for available cameras
        res = self._list_svc.call(roslibpy.ServiceRequest({}))
        if self._id not in res.get('cameras', []):
            raise AttributeError(f"Cannot locate a service for camera name '{self._id}'. Close a different camera first.")

        self._settings = {
            'width': 320,
            'height': 200,
            'fps': 20,
            'controls': []
        }
        self._open = False

    def _reload(self):
        self.open()

    def _set_control_value(self, control_id, value):
        for c in self._settings['controls']:
            if c['id'] == control_id:
                c['value'] = value
                return
        self._settings['controls'].append({'id': control_id, 'value': value})

    @property
    def resolution(self):
        return (self._settings['width'], self._settings['height'])

    @resolution.setter
    def resolution(self, res):
        res = tuple(res)
        if res not in self.MODES:
            raise ValueError(f"Invalid camera mode {res[0]}x{res[1]}")
        self._settings['width'] = res[0]
        self._settings['height'] = res[1]
        self._reload()

    def open(self):
        if self._id == 'head_camera':
            self._set_control_value(2, True) # CAMERA_CONTROL_FLIP
            self._set_control_value(3, True) # CAMERA_CONTROL_MIRROR
            
        req = roslibpy.ServiceRequest({'name': self._id, 'settings': self._settings})
        ret = self._open_svc.call(req)
        
        if ret.get('err', 0) != 0:
            raise OSError(ret['err'], "Failed to open camera")
        self._open = True

    def close(self):
        req = roslibpy.ServiceRequest({'name': self._id})
        ret = self._close_svc.call(req)
        if ret.get('err', 0) != 0 and ret.get('err', 0) != errno.EINVAL:
            raise OSError(ret['err'], "Failed to close camera")
        self._open = False