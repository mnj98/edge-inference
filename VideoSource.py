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

        ret, jpeg = cv2.imencode('.jpg', image)
        #print(jpeg)
        #print(jpeg.tobytes())
        self.current_image_id += 1
        return (self.current_image_id - 1, jpeg.tobytes())


def request_inference(image, id, index, result_buffer, model = 'mobilenet', url='http://localhost:1234/infer'):
    req = requests.post(url, files = {'image': image}, data = {'id': id, 'model': model})
    top_result = literal_eval(req.content.decode())[0]

    result_buffer[index] = top_result


def main():
    images = Source()
    num_to_test = 1000
    fps = 30
    frame_delay = 1 / fps 
    truths_file = open('/home/pi/ImageNet/2012/2012_ground_truth_ids.txt', 'r')
    true_classes = np.ndarray(shape=(num_to_test,), dtype='int32')
    inf_classes = np.ndarray(shape=(num_to_test,), dtype='int32')

    for i in range(num_to_test):
        true_classes[i] = int(truths_file.readline()[1:])
    print(true_classes)
    print(inf_classes)



    
    #image_id = int(random.random() * 1000)

    start_time = time.time()

    for i in range(num_to_test):
        image_id, image = images.get_frame()
        threading.Thread(target=request_inference, args=(image, image_id, i, inf_classes)).start()
        time.sleep(frame_delay - ((time.time() - start_time) % frame_delay))


    print(inf_classes)
    '''
    start = time.time()
    req = requests.post('http://localhost:1234/infer', \
        files = {'image': image}, data = {'id': image_id, 'model': 'mobilenet'})#data={'model': 'mobilenet', 'image': images.get_frame()})
    print(req.content, time.time() - start)
    '''


if __name__ == '__main__':
   main()
