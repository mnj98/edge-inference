import cv2, copy, time, configparser, csv
from threading import Lock
from simple_pid import PID

on_pi = True

images_path = "/home/pi/ImageNet/2012/val/ILSVRC2012_val_%08d.JPEG" if on_pi else \
    "/Users/mnj98/ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG"


class Config(object):
    def __init__(self, file):
        parser = configparser.ConfigParser()
        parser.read(file)
        
        self.samples = parser.getint('Video Source', 'samples')
        self.model = parser.get('Video Source', 'model')
        self.timeout = parser.getfloat('Video Source', 'latency_timeout')
        self.ofps = parser.getint('Controller', 'initial_offloading_rate')
        self.sfps = parser.getint('Video Source', 'fps')
        self.mps = parser.getfloat('Controller', 'measure_rate')
        self.enabled = parser.getboolean('Controller', 'enable_offloading')
        size = parser.getint('Video Source', 'size')
        self.shape = (size, size)
        [self.p, self.i, self.d] = [parser.getfloat('Controller', j) for j in ['p', 'i', 'd']]
        self.set_point = parser.getfloat('Controller', 'set_point')
        self.PID_enabled = parser.getboolean('Controller', 'enable_pid')

        self.proc_count = 0
        self.timeout_count = 0

        self.pps = []
        self.tps = []

        n = open(parser.get('Network', 'file'), 'r', newline='')
        net_stats = list(csv.DictReader(n))
        n.close()
        self.net_stats = net_stats


        self.locks = {'fps': Lock(),\
            'mps': Lock(),\
            'enabled': Lock(),\
            'proc': Lock(),\
            'timeout': Lock(),\
            'pps': Lock(),\
            'tps': Lock()}
    def get_network_conditions(self):return self.net_stats
    def is_PID_enabled(self): return self.PID_enabled
    def get_p(self): return self.p
    def get_i(self): return self.i
    def get_d(self): return self.d
    def get_set_point(self): return self.set_point
    def get_shape(self): return self.shape
    def get_samples(self): return self.samples
    def get_model(self): return self.model
    def get_latency_timeout(self): return self.timeout
    def get_source_fps(self): return self.sfps
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

class Offload_Controller(object):
    def __init__(self, config: Config):
        self.config = config
        self.controller = PID(self.config.get_p(),\
            self.config.get_i(),\
            self.config.get_d(),\
            setpoint=self.config.get_set_point())
        self.controller.output_limits = (0, self.config.get_source_fps())
        self.controller.sample_time = None
    
    def control_and_update(self, tps):
        if tps == None or not self.config.is_PID_enabled(): return
        ofps = self.config.get_offload_fps()
        ratio = ofps / (ofps - tps + 1)
        new_ofps = round(self.controller(ratio))
        if new_ofps == 0: new_ofps = 1
        print('new ofps:', new_ofps)
        self.config.set_offload_fps(new_ofps)

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
