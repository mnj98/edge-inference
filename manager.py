#from asyncio import base_tasks
#import  multiprocessing
#from multiprocessing import Process
#from random import random
#from time import sleep
#from server import run
import numpy as np
import queue
import threading
import random
import time
import sys


from flask import Flask, render_template, Response, request
import cv2

def flask_thread():
    
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

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
        #print(res)
        r = results.pop(image_id)
        #print(r) 
        return r
    
    app.run(host='0.0.0.0', threaded=True,  port=1234)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.applications import MobileNetV3Large, EfficientNetB0
#from tensorflow.keras.preprocessing import image
import tensorflow.keras.applications.mobilenet_v3 as mobilenet
import tensorflow.keras.applications.efficientnet as efficientnet


keras_model_names = ['mobilenet', 'efficientnet']
processing_functions = {'mobilenet': mobilenet.preprocess_input, 'efficientnet': efficientnet.preprocess_input}
decode_functions = {'mobilenet': mobilenet.decode_predictions, 'efficientnet': efficientnet.decode_predictions}
models = {'mobilenet': MobileNetV3Large(weights='imagenet'), 'efficientnet': EfficientNetB0(weights='imagenet')}
requests = {'mobilenet': queue.Queue(), 'efficientnet': queue.Queue()}
events = queue.Queue()
results = dict()

NUM_EVENTS = 3000
if len(sys.argv) == 2:
    BATCH_SIZE = int(sys.argv[1])
else:
    BATCH_SIZE = 10

def inference_thread(model_name):
    local = threading.local()
    local.batch_n = 0
    while True:
        print('batch number:', local.batch_n)
        local.batch_n += 1
        local.batch = []
        #cut_off_time = time.time()
        local.idx = 0
        while local.idx < BATCH_SIZE:
            try:
                local.r = requests[model_name].get(timeout=0.1)
                local.batch.append(local.r)
                local.idx += 1
            except:
                if len(local.batch) == 0:
                    local.idx = 0
                else:
                    break
                
        '''for i in range(BATCH_SIZE):
            try:
                r = requests.get(timeout=1)
                batch.append(r)
            except:
                if len(batch) == 0:
        '''

        local.batch_images = np.array(list(map(lambda img: img['image'], local.batch)))
        local.batch_events = list(map(lambda img: img['done_event'], local.batch))
        local.batch_ids = list(map(lambda img: img['id'], local.batch))

        local.frames = processing_functions[model_name](local.batch_images)
        local.preds = decode_functions[model_name](local.model.predict(local.frames, verbose=1), top = 5)
        #print('preds',preds)
        #print(decode_predictions(preds, top=5))

        for i in range(len(local.batch)):
            results[local.batch_ids[i]] = list(map(lambda pr: int(pr[0][1:]),local.preds[i]))
        for e in local.batch_events:
            e.set()
            #events.put(manager.Event())

def main():
    for i in range(NUM_EVENTS):
        events.put(threading.Event())

    keras_model_threads = [threading.Thread(target=inference_thread, args=(i,)) for i in keras_model_names]
    

    server_thread = threading.Thread(target=flask_thread, args=())

    for thread in keras_model_threads:
        thread.start()
    server_thread.start()

if __name__ == '__main__':
    main()

'''




def qpush(req, res, events):
    run(req, res, events)



with multiprocessing.Manager() as manager:
    requests = manager.Queue()
    results = manager.dict()
    events = manager.Queue()

    for i in range(1000):
        events.put(manager.Event())

    server = multiprocessing.Process(target=qpush, args=(requests, results, events))

    server.start()

    batch_size = 10
    while True:
        batch = []
        for i in range(batch_size):
            r = requests.get()
            batch.append(r)
        
        #image = r['image']
        batch_images = np.array(list(map(lambda img: img['image'], batch)))
        batch_events = list(map(lambda img: img['done_event'], batch))
        batch_ids = list(map(lambda img: img['id'], batch))
        #frame = np.expand_dims(image, axis=0)
        frames = preprocess_input(batch_images)
        preds = decode_predictions(model.predict(frames, verbose=1), top = 5)
        #print('preds',preds)
        #print(decode_predictions(preds, top=5))

        for i in range(batch_size):
            results[batch_ids[i]] = list(map(lambda pr: int(pr[0][1:]),preds[i]))
        for e in batch_events:
            e.set()
            events.put(manager.Event())

'''