import cv2, time, requests, threading, sys, queue
from ast import literal_eval
from http.client import RemoteDisconnected
import numpy as np

class Source(object):
    def __init__(self, shape):
        self.shape = shape
        self.video = cv2.VideoCapture("/home/pi/ImageNet/2012/val/ILSVRC2012_val_%08d.JPEG")#("/Users/mnj98/Desktop/ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG")

    def __del__(self):
        self.video.release()

    def get_frame(self, index = None):
        if index != None:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
        frame_index = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        success, image = self.video.read()

        ret, jpeg = cv2.imencode('.jpg', cv2.resize(image, self.shape))
        return (frame_index, jpeg.tobytes())


def request_inference( image, index, result_buffer, inference_times, model = 'mobilenet', url='http://localhost:1234/infer'):
    try:
        t = time.time()
        #print(id, t)
        req = requests.post(url, files = {'image': image}, data = {'model': model})
        top_result = literal_eval(req.content.decode())[0]
        inf_time = time.time() - t
        #print(inf_time)
        #print('index:', index, 'true:', true_classes[index], 'res:', top_result, true_classes[index] == top_result, 'time:', inf_time)
        result_buffer[index] = top_result
        inference_times[index] = inf_time
    except RemoteDisconnected:
        print('remote disconnected error on index:', index)
    except ConnectionError:
        print('connection error on index:', index)
    except:
        print('other error on index:', index)


def capture_loop(q, num_to_test, shape):
    images = Source()

    for i in range(num_to_test):
        q.put(images.get_frame())

def main(num_to_test, fps, model):
    if model == 'efficient_det':
        shape = (500,500)
    else:
        shape = (224,224)
    frame_delay = 1/fps
    truths_file = open('/home/pi/ImageNet/2012/2012_ground_truth_ids.txt', 'r')
    true_classes = np.ndarray(shape=(num_to_test,), dtype='int32')
    inf_classes = np.ndarray(shape=(num_to_test,), dtype='int32')
    times = np.ndarray(shape=(num_to_test,))

    for i in range(num_to_test):
        true_classes[i] = int(truths_file.readline()[1:])

    image_queue = queue.Queue(maxsize=180)
    capture_thread = threading.Thread(target=capture_loop, args=(image_queue, num_to_test, shape))

    threads = np.ndarray(shape=(num_to_test,), dtype=threading.Thread)
    capture_thread.start()
    time.sleep(0.1)
    #capture_thread.join()
    start_time = time.time()
    for i in range(num_to_test):
        #print(image_queue.qsize())
        image_id, image = image_queue.get()
        thread = threading.Thread(target=request_inference, args=(image, image_id, inf_classes, times, model))
        threads[i] = thread
        thread.start()
        wait = frame_delay - ((time.time() - start_time) % frame_delay)
        time.sleep(wait)

    for thread in threads:
        thread.join()
    capture_thread.join()
    print('accuracy:',np.sum(inf_classes == true_classes) / num_to_test)
    print('fps measured:', num_to_test / (time.time() - start_time), 'input:', fps)
    print('inf latency:', np.sum(times) / num_to_test)

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
