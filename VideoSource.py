import requests
import cv2
import random
import time
import numpy as np
from ast import literal_eval

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





def main():
    images = Source()
    num_to_test = 1000
    truths_file = open('/home/pi/ImageNet/2012/2012_ground_truth_ids.txt', 'r')
    true_classes = np.ndarray(shape=(num_to_test,), dtype='int32')
    inf_classes = np.ndarray(shape=(num_to_test,), dtype='int32')

    for i in range(num_to_test):
        true_classes[i] = int(truths_file.readline()[1:])
    print(true_classes)
    print(inf_classes)
    x = 3 / 0 
    image_id, image = images.get_frame()
    image_id = int(random.random() * 1000)
    start = time.time()
    req = requests.post('http://localhost:1234/infer', \
        files = {'image': image}, data = {'id': image_id, 'model': 'mobilenet'})#data={'model': 'mobilenet', 'image': images.get_frame()})
    print(req.content, time.time() - start)


if __name__ == '__main__':
   main()
