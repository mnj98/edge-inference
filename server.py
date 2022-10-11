def run(req, res, events):
    from flask import Flask, render_template, Response, request
    import random
    import cv2
    import numpy as np
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.post('/infer')
    def inference():
        image = request.files['image'].read()
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        id = random.random()
        e = events.get()
        req.put({'id': id, 'done_event': e, 'image': image})
        e.wait()
        r = res.pop(id)
        print(r)
        return '<p>OK!</p>'

    app.run(host='0.0.0.0', threaded=True,  port=1234)
