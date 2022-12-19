import cv2, time, requests, threading, sys, queue
from ast import literal_eval
from http.client import RemoteDisconnected
import numpy as np
import argparse

class Timeouts(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
    def add(self, count = 1):
        with self.lock:
            self.count += count
    def get(self):
        with self.lock:
            return self.count


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


def request_inference( image, index, result_buffer, inference_times, timeouts, model = 'mobilenet', url='http://localhost:1234/infer'):
    try:
        t = time.time()
        #print(id, t)
        req = requests.post(url, files = {'image': image}, data = {'model': model}, timeout=1)
        if model != 'efficient_det':
            data = literal_eval(req.content.decode())
            top_result = data[0]
            p_time = data[-1]
        else:
            top_result = req.content.decode()
        inf_time = time.time() - t
        print("Total time:",inf_time, "processing time:", p_time)
        #print(top_result)
        #print('index:', index, 'true:', true_classes[index], 'res:', top_result, true_classes[index] == top_result, 'time:', inf_time)
        if model != 'efficient_det':
            result_buffer[index] = top_result
        inference_times[index] = inf_time
    except RemoteDisconnected:
        print('remote disconnected error on index:', index)
    except ConnectionError:
        print('connection error on index:', index)
    except requests.exceptions.ReadTimeout as e:
        print('timeout')
        timeouts.add()
    #except:
        #print('other error on index:', index)


def capture_loop(q, num_to_test, shape):
    images = Source(shape)

    for i in range(num_to_test):
        q.put(images.get_frame())

def main(args):#(num_to_test, fps, model):
    timeouts = Timeouts()
    num_to_test = args.numframes
    if args.model == 'efficient_det':
        shape = (500,500)
    else:
        shape = (224,224)
    frame_delay = 1/args.fps
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
        thread = threading.Thread(target=request_inference, args=(image, image_id, inf_classes, times, timeouts, args.model))
        threads[i] = thread
        thread.start()
        wait = frame_delay - ((time.time() - start_time) % frame_delay)
        time.sleep(wait)

    for thread in threads:
        thread.join()
    capture_thread.join()

    if not args.test:
        if args.model != 'efficient_det':
            print('accuracy:',np.sum(inf_classes == true_classes) / num_to_test)
        print('fps measured:', (num_to_test - timeouts.get())/ (time.time() - start_time), 'input:', args.fps)
        print('inf latency:', np.sum(times) / num_to_test)
        print('timeouts:', timeouts.get())
    else:
        results_file = open(time.strftime('%Y-%m-%d-%H-%M-%S') ,'w')
        results_file.write(str(args) + '\n')
        results_file.write('Average latency: ' + str(np.sum(times) / num_to_test) + '\n')
        results_file.write('Latencies: ' + str(times) + '\n')
        if args.model != 'efficient_det':
            results_file.write('accuracy: ' + str(np.sum(inf_classes == true_classes) / num_to_test) + '\n')
        results_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Source images for remote inference")
    parser.add_argument('-n', '--numframes' , type=int, required=True)
    parser.add_argument('-f', '--fps', type=float, required=True)
    parser.add_argument('-m', '--model', choices=['mobilenet', 'efficientnet', 'efficient_det'], default='mobilenet')
    parser.add_argument('-t', '--test', type=int, default=0, help='include an int that represents the batch size on the backend')

    main(parser.parse_args())
