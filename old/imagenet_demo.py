import cv2 as cv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
import numpy as np
import sys
model = MobileNetV3Large(weights='imagenet')
num_to_test = 500


images = cv.VideoCapture("/home/mnj98/ImageNet/2012/val/ILSVRC2012_val_%08d.JPEG")
truths_file = open("/home/mnj98/ImageNet/2012/2012_ground_truth_ids.txt", "r")
truths = truths_file.read().split('\n')
predicted = []
num_correct = 0
for i in range(num_to_test):
    _, frame = images.read()
    #frame = cv.resize(frame, (224,224))
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    preds = model.predict(frame, verbose=1)
    #for pred in preds:
    #    top_indices = pred.argsort()[-5:][::-1]
    #    print(top_indices)
    #print(np.argmax(preds))
    #print(truths[i])
    #print('Predicted:', decode_predictions(preds, top=1)[0][0][0])
    #predicted.append(decode_predictions(preds, top=1)[0][0][0])
    if decode_predictions(preds, top=1)[0][0][0] == truths[i]:
        num_correct += 1

print(num_correct / num_to_test)
#correct = truths[:num_to_test - len(truths)] == predicted
#print(correct)
truths_file.close()
