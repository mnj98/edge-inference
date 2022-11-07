import argparse
parser = argparse.ArgumentParser(description="Server for remote inference")
parser.add_argument('-b', '--batchsize' , type=int, required=True)
parser.add_argument('-d', '--debug', action='store_true')
args = parser.parse_args()

import numpy as np
import queue
import threading
import random
import time
if not args.debug:
    import tensorflow_hub as hub
    import tensorflow as tf

from flask import Flask, render_template, request
import cv2

import logging
logging.getLogger('werkzeug').disabled = True


def flask_thread():
    
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')
    @app.route('/ping')
    def ping():
        return 'pong'


    @app.post('/infer')
    def inference():
        image_id = random.random() #request.values['id']
        model_to_use = request.values['model']
        image = request.files['image'].read()
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    
        e = events.get()
        requests[model_to_use].put({'id': image_id, 'done_event': e, 'image': image})
        e.wait()
        e.clear()
        events.put(e)
        r = results.pop(image_id)
        return r
    
    app.run(host='0.0.0.0', threaded=True,  port=1234)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if not args.debug:
    from tensorflow.keras.applications import  EfficientNetB0
from tensorflow.keras.applications import MobileNetV3Large
import tensorflow.keras.applications.mobilenet_v3 as mobilenet
if not args.debug:
    import tensorflow.keras.applications.efficientnet as efficientnet

if not args.debug:
    efficient_det_model = 'https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1'

if not args.debug:
    keras_model_names = ['mobilenet', 'efficientnet']
    processing_functions = {'mobilenet': mobilenet.preprocess_input, 'efficientnet': efficientnet.preprocess_input}
    decode_functions = {'mobilenet': mobilenet.decode_predictions, 'efficientnet': efficientnet.decode_predictions}
    models = {'mobilenet': MobileNetV3Large(weights='imagenet'), 'efficientnet': EfficientNetB0(weights='imagenet'),\
            'efficient_det': hub.load(efficient_det_model)}
    requests = {'mobilenet': queue.Queue(), 'efficientnet': queue.Queue(), 'efficient_det': queue.Queue()}
else:
    keras_model_names = ['mobilenet']
    processing_functions = {'mobilenet': mobilenet.preprocess_input}
    decode_functions = {'mobilenet': mobilenet.decode_predictions}
    models = {'mobilenet': MobileNetV3Large(weights='imagenet')}
    requests = {'mobilenet': queue.Queue()}

events = queue.Queue()
results = dict()

NUM_EVENTS = 3000

def det_thread(model_name, BATCH_SIZE):
    batch_n = 0
    while True:
        print(model_name,': batch number:', batch_n)
        batch_n += 1
        batch = []

        idx = 0
        while idx < BATCH_SIZE:
            try:
                r = requests[model_name].get(timeout=0.1)
                batch.append(r)
                idx += 1
            except:
                if len(batch) == 0:
                    idx = 0
                else:
                    break
        batch_images = tf.constant(np.array(list(map(lambda img: img['image'], batch))))
        batch_events = list(map(lambda img: img['done_event'], batch))
        batch_ids = list(map(lambda img: img['id'], batch))

        boxes, scores, classes, num_detections = models['efficient_det'](batch_images)

        for i in range(len(batch)):
            results[batch_ids[i]] = 'ok!'#(boxes[i], scores[i], classes[i], num_detections[i])
            batch_events[i].set()
local = threading.local()
BATCH_TIME_ESTIMATE = 0.025
def inference_thread(model_name, BATCH_SIZE):
    #local = threading.local()
    local.batch_n = 0
    local.prev_batch_size = 0
    while True:
        local.t = time.time()
        print(model_name,': batch number:', local.batch_n, 'size:', local.prev_batch_size)
        local.batch_n += 1
        local.batch = []
        #cut_off_time = time.time()
        local.batch.append(requests[model_name].get(True))
        local.idx = 1
        local.q_size = requests[model_name].qsize()
        local.batch_collect_start_time = time.time()
        for i in range(local.q_size): #while (time.time() - local.batch_collect_start_time) < BATCH_TIME_ESTIMATE or local.idx == 0: #while local.idx < BATCH_SIZE:
            if local.idx > 0: print(time.time() - local.batch_collect_start_time)
            try:
                local.r = requests[model_name].get(False) #(timeout=BATCH_TIME_ESTIMATE + 0.005)
                local.batch.append(local.r)
                local.idx += 1
            except:
                if len(local.batch) == 0:
                    local.idx = 0
                else:
                    break

        if len(local.batch) == 0: continue
        local.prev_batch_size = len(local.batch)
        local.batch_images = np.array(list(map(lambda img: img['image'], local.batch)))
        local.batch_events = list(map(lambda img: img['done_event'], local.batch))
        local.batch_ids = list(map(lambda img: img['id'], local.batch))

        local.frames = processing_functions[model_name](local.batch_images)
        local.pred_time = time.time()
        local.preds = decode_functions[model_name](models[model_name].predict_on_batch(local.frames), top = 5)
        print('pred_time', time.time() - local.pred_time)
        #print('preds',preds)
        #print(decode_predictions(preds, top=5))

        for i in range(len(local.batch)):
            results[local.batch_ids[i]] = list(map(lambda pr: int(pr[0][1:]),local.preds[i]))
            local.batch_events[i].set()
        print('batch time:', time.time() - local.t)

def main():
    for i in range(NUM_EVENTS):
        events.put(threading.Event())
    if args.debug:
        print('----- debug mode -----')
        mob_thread = threading.Thread(target=inference_thread, args=('mobilenet', args.batchsize,))
        server_thread = threading.Thread(target=flask_thread, args=())
        mob_thread.start()
        server_thread.start()

    else:
        keras_model_threads = [threading.Thread(target=inference_thread, args=(i,args.batchsize,)) for i in keras_model_names]
    
        detection_thread = threading.Thread(target=det_thread, args=('efficient_det',args.batchsize,))

        server_thread = threading.Thread(target=flask_thread, args=())
        detection_thread.start()
        for thread in keras_model_threads:
            thread.start()
        server_thread.start()
         

if __name__ == '__main__':
    main()
