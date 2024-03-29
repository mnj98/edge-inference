import tensorflow.keras.applications as models
import cv2, time
import numpy as np
from Helpers import inf_response as Res

def infer_loop(in_q, out_q, ready_event, pull_event, model_name = 'mobilenet'):
    if model_name == 'mobilenet':
        model = models.MobileNetV3Small(weights='imagenet')
        preprocess = models.mobilenet_v3.preprocess_input
        decode = models.mobilenet_v3.decode_predictions
    elif model_name == 'efficientnet':
        model = models.EfficientNetB0(weights='imagenet')
        preprocess = models.efficientnet.preprocess_input
        decode = models.efficientnet.decode_predictions
    else:
        raise Exception("bad model")


    ready_event.set()
    while True:
        pull_event.wait()
        ready_event.clear()
        req = in_q.get()
        t = time.time()
        frame = cv2.imdecode(np.frombuffer(req.image, np.uint8), cv2.IMREAD_COLOR)
        frame = preprocess(np.expand_dims(frame, axis=0))

        classification = model.predict_on_batch(frame)
        preds = list(map(lambda pr: int(pr[0][1:]), decode(classification, top = 5)[0]))
        lat = time.time() - t
        out_q.put(Res(req.id, preds, True, lat))
        pull_event.clear()
        ready_event.set()
