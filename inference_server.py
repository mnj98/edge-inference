from flask import Flask, render_template, Response, request
import cv2
import numpy as np


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions

model = MobileNetV3Large(weights='imagenet')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.post('/infer')
def inference():
    #print(request.values)
    #for value in request.values:
    #     print(value)
    #print(type(request.values['image']))
    image = request.files['image'].read() #.encode('utf-8')
    #print(type(image.read()))
    image = np.frombuffer(image, np.uint8)
    print(image)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #print(image)
    #cv2.imwrite('image--.jpeg', image)

    frame = np.expand_dims(image, axis=0)
    frame = preprocess_input(frame)
    preds = model.predict(frame, verbose=1)
    print(decode_predictions(preds, top=5))



    return('<p>OK!</p>')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=1234)
