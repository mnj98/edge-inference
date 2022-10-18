import sys
import requests
import cv2
import random
import time
import numpy as np
from ast import literal_eval
import threading

class Source(object):
    def __init__(self):
        self.video = cv2.VideoCapture("/home/pi/ImageNet/2012/val/ILSVRC2012_val_%08d.JPEG")
        self.current_image_id = 1

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        ret, jpeg = cv2.imencode('.jpg', cv2.resize(image, (224,224)))
        #print(jpeg)
        #print(jpeg.tobytes())
        self.current_image_id += 1
        return (self.current_image_id - 1, jpeg.tobytes())


def request_inference(true_classes, image, id, index, result_buffer, model = 'mobilenet', url='http://localhost:1234/infer'):
    req = requests.post(url, files = {'image': image}, data = {'id': id, 'model': model})
    top_result = literal_eval(req.content.decode())[0]
    #print('index:', index, 'true:', true_classes[index], 'res:', top_result, true_classes[index] == top_result)
    result_buffer[index] = top_result


def main(args):
    images = Source()
    if len(args) == 2:
        num_to_test = int(args[1])
    else:
        num_to_test = 300
    fps = 30
    frame_delay = 1 / fps 
    truths_file = open('/home/pi/ImageNet/2012/2012_ground_truth_ids.txt', 'r')
    true_classes = np.ndarray(shape=(num_to_test,), dtype='int32')
    inf_classes = np.ndarray(shape=(num_to_test,), dtype='int32')

    for i in range(num_to_test):
        true_classes[i] = int(truths_file.readline()[1:])
    #print(true_classes)
    #print(inf_classes)



    
    #image_id = int(random.random() * 1000)
    t = time.time()
    start_time = time.time()
    threads = []
    for i in range(num_to_test):
        image_id, image = images.get_frame()
        thread = threading.Thread(target=request_inference, args=(true_classes, image, image_id, i, inf_classes))
        threads.append(thread)
        thread.start()
        time.sleep(frame_delay - ((time.time() - start_time) % frame_delay))

    for thread in threads:
        thread.join()
    print(np.sum(inf_classes == true_classes) / num_to_test)
    print(num_to_test / (time.time() - t))
    '''
    start = time.time()
    req = requests.post('http://localhost:1234/infer', \
        files = {'image': image}, data = {'id': image_id, 'model': 'mobilenet'})#data={'model': 'mobilenet', 'image': images.get_frame()})
    print(req.content, time.time() - start)
    '''


if __name__ == '__main__':
   main(sys.argv)
