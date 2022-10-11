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
    print('qpush')
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
    print(procs[0].pid)
    while True:
        r = requests.get()
        print('image', r)
        image = r['image']
        #print("waiting")
        #sleep(1)


        frame = np.expand_dims(image, axis=0)
        frame = preprocess_input(frame)
        preds = model.predict(frame, verbose=1)
        print(decode_predictions(preds, top=5))


        results[r['id']] = 'Hello!!'
        r['done_event'].set()
        events.put(manager.Event())

#    for proc in procs:
#        proc.join()
#    for j in range(3):
#        print('qget')
#        print(q.get())
#
#    for proc in procs:
#        proc.join()




#manager = SyncManager(address=("", 12345), authkey=b'1234')
#manager.get_server().serve_forever()
#data_queue = manager.Queue()

#data_queue.put("Hello World")
#print("done")
