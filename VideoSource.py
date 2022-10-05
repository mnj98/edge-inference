import requests
import cv2

class Source(object):
    def __init__(self):
        self.video = cv2.VideoCapture("/home/mnj98/ImageNet/2012/val/ILSVRC2012_val_%08d.JPEG")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        ret, jpeg = cv2.imencode('.jpg', image)
        print(jpeg)
        #print(jpeg.tobytes())
        return jpeg.tobytes()





def main():
    images = Source()
    req = requests.post('http://localhost:1234/infer', \
        files = {'image': images.get_frame()})#data={'model': 'mobilenet', 'image': images.get_frame()})



if __name__ == '__main__':
   main()
