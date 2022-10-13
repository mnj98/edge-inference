from asyncio import base_tasks
import  multiprocessing
#from multiprocessing import Process
from random import random
from time import sleep
from server import run
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions

model = MobileNetV3Large(weights='imagenet')




def qpush(i, req, res, events):
    run(req, res, events)



with multiprocessing.Manager() as manager:
    requests = manager.Queue()
    results = manager.dict()
    events = manager.Queue()

    for i in range(100):
        events.put(manager.Event())
    i = 1
    procs = [multiprocessing.Process(target=qpush, args=(j, requests, results, events)) for j in range(i)]

    for proc in procs:
        proc.start()

    batch_size = 1
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
        print('preds',preds)
        #print(decode_predictions(preds, top=5))

        for i in range(batch_size):
            results[batch_ids[i]] = list(map(lambda pr: int(pr[0][1:]),preds[i]))
        for e in batch_events:
            e.set()
            events.put(manager.Event())

