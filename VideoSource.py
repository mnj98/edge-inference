import cv2

on_pi = False

images_path = "/home/pi/ImageNet/2012/val/ILSVRC2012_val_%08d.JPEG" if on_pi else \
    "/Users/mnj98/Desktop/ILSVRC2012_img_val/ILSVRC2012_val_%08d.JPEG"

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
        return (frame_index, jpeg.tobytes())