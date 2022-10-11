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

    while True:
        r = requests.get()
        image = r['image']

        frame = np.expand_dims(image, axis=0)
        frame = preprocess_input(frame)
        preds = model.predict(frame, verbose=1)
        print(decode_predictions(preds, top=5))


        results[r['id']] = 'Hello!!'
        r['done_event'].set()
        events.put(manager.Event())

