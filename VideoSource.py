import requests
import cv2

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

    image_id, image = images.get_frame()
    req = requests.post('http://localhost:1234/infer', \
        files = {'image': image}, data = {'id': image_id, 'model': 'mobilenet'})#data={'model': 'mobilenet', 'image': images.get_frame()})
    print(req.json())


if __name__ == '__main__':
   main()
