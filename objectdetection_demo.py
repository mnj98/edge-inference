import cv2 as cv
import tensorflow_hub as hub
from PIL import Image
from six import BytesIO
import numpy as np
import tensorflow as tf
#https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1
efficient_det_model = 'https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1'
yolo_model = 'https://tfhub.dev/rishit-dagli/yolo-cppe5/1' # requires 800x1216 with values in range 0-1
#model_info = ('SSD MobileNet v2 320x320' , 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')
import time

def main():
    model = hub.load(efficient_det_model)
    #print(model)

    images = cv.VideoCapture("/home/mnj98/ImageNet/2012/val/ILSVRC2012_val_%08d.JPEG")
    _, image = images.read()
    print(model(np.expand_dims(image, axis=0)))
    for i in range(10):
        frames = []
        for i in range(10):
            _, image = images.read()
            start = time.time()
            frames.append(tf.constant(cv.resize(image, (500,500))))
        frames = tf.constant(np.array(frames))
        print(type(frames))
        #x = 3 / 0 #exit script
        results = model(frames) #load_image_into_numpy_array("/home/mnj98/ImageNet/2012/val/ILSVRC2012_val_00000001.JPEG"))
        print('time:', time.time() - start)




def load_image_into_numpy_array(path):
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    image_data = cv.imread(path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)










COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
 (0, 2),
 (1, 3),
 (2, 4),
 (0, 5),
 (0, 6),
 (5, 7),
 (7, 9),
 (6, 8),
 (8, 10),
 (5, 6),
 (5, 11),
 (6, 12),
 (11, 12),
 (11, 13),
 (13, 15),
 (12, 14),
 (14, 16)]


if __name__ == "__main__":
    main()
