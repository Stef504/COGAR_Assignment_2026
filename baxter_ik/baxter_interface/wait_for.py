import time

def wait_for(test, timeout=1.0, raise_on_error=True, rate=100, timeout_msg="timeout expired", body=None):
    end_time = time.time() + timeout
    sleep_duration = 1.0 / rate
    notimeout = (timeout < 0.0) or timeout == float("inf")
    
    while not test():
        if not notimeout and time.time() >= end_time:
            if raise_on_error:
                raise TimeoutError(timeout_msg)
            return False
        if callable(body):
            body()
        time.sleep(sleep_duration)
    return True

class Signal(object):
    def __init__(self):
        self._functions = set()
    def __call__(self, *args, **kargs):
        for f in self._functions: f(*args, **kargs)
    def connect(self, slot): self._functions.add(slot)
    def disconnect(self, slot): self._functions.discard(slot)