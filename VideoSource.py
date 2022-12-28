import cv2, copy, time
from threading import Lock

on_pi = True

images_path = "/home/pi/ImageNet/2012/val/ILSVRC2012_val_%08d.JPEG" if on_pi else \
    "/Users/mnj98/ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG"


class Config(object):
    def __init__(self, source_fps, offload_fps, mps=1, enabled=True, shape=(224,224)):
        self.locks = {'fps': Lock(),\
            'mps': Lock(),\
            'enabled': Lock(),\
            'proc': Lock(),\
            'timeout': Lock(),\
            'pps': Lock(),\
            'tps': Lock()}
        self.ofps = offload_fps
        self.sfps = source_fps
        self.mps = mps
        self.enabled = enabled
        self.shape = shape

        self.proc_count = 0
        self.timeout_count = 0

        self.pps = []
        self.tps = []
    def measure_and_report_fps(self):
        with self.locks['pps']:
            with self.locks['proc']:
                if len(self.pps) < 1:
                    self.pps.append((self.proc_count, time.time()))
                    return None
                delta_p = self.proc_count - self.pps[-1][0]
                t = time.time()
                delta_t = t - self.pps[-1][1]
                fps = delta_p / delta_t
                self.pps.append((self.proc_count, t))
                return fps
    def measure_and_report_tps(self):
        with self.locks['tps']:
            with self.locks['timeout']:
                if len(self.tps) < 1:
                    self.tps.append((self.timeout_count, time.time()))
                    return None
                delta_timeout = self.timeout_count - self.tps[-1][0]
                t = time.time()
                delta_t = t - self.tps[-1][1]
                tps = delta_timeout / delta_t
                self.tps.append((self.timeout_count, t))
                return tps

    def get_proc_rates(self):
        with self.locks['pps']:
            return copy.deepcopy(self.pps)
    def get_timeout_rates(self):
         with self.locks['tps']:
             return copy.deepcopy(self.tps)
    def disable_offloading(self):
        with self.locks['enabled']:
            self.enabled = False
    def enable_offloading(self):
        with self.locks['enabled']:
            self.enabled = True
    def is_offloading_enabled(self):
        with self.locks['enabled']:
            return self.enabled
    def get_measure_rate(self):
        with self.locks['mps']:
            return self.mps
    def set_measure_rate(self, n = 1):
        with self.locks['mps']:
            self.mps = n
    def set_offload_fps(self, n = None):
        if n == None:
            n = self.sfps
        with self.locks['fps']:
            self.ofps = n
    def get_offload_fps(self):
        with self.locks['fps']:
            return self.ofps
    def add_proc(self, n = 1):
        with self.locks['proc']:
            self.proc_count += n
    def add_timeout(self, n = 1):
        with self.locks['timeout']:
            self.timeout_count += n
    def get_procs(self):
        with self.locks['proc']:
            return self.proc_count
    def get_timeouts(self):
        with self.locks['timeout']:
            return self.timeout_count

class VideoSource(object):
    def __init__(self, shape):
        self.shape = shape
        self.video = cv2.VideoCapture(images_path)

    def __del__(self):
        self.video.release()

    def get_frame(self, index = None):
        if index != None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        frame_index = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        success, image = self.video.read()

        ret, jpeg = cv2.imencode('.jpg', cv2.resize(image, self.shape))
        return inf_request(frame_index, jpeg.tobytes())


class inf_response(object):
    def __init__(self, id, classes, local, latency = None, success = True):
        self.id = id
        self.classes = classes
        self.local = local
        self.latency = latency
        self.success = success

class inf_request(object):
    def __init__(self, id, image, model = 'mobilenet'):
        self.id = id
        self.image = image
        self.model = model
