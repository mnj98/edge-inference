import cv2 as cv
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import sys
import time
import socket, struct
from io import BytesIO
model = MobileNetV3Large(weights='imagenet')

#if len(sys.argv) < 2:
#    print("Please specify image")
#img_path = sys.argv[1]
#img_path = 'elephant.jpg'
#img = image.load_img(img_path, target_size=(224, 224))
#x = image.img_to_array(img)



#s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

#hostname = socket.gethostname()
#host_ip = socket.gethostbyname(hostname)
port = 1234
socket_addr = (sys.argv[1],port)
#s.bind(socket_addr)
s = socket.create_connection(socket_addr)
print(socket_addr)
payload_size = struct.calcsize("L")
#s.listen(10)


print("waiting for connection")
#conn, addr = s.accept()
print('accepted connection')
count = 0
num_sent_frames = 0
num_frames_to_send = int(sys.argv[2])
while num_sent_frames < num_frames_to_send:
    num_sent_frames += 1
    data = s.recv(payload_size)

    if data:
        data_size = struct.unpack("L", data)[0]

        data = b''

        while len(data) < data_size:
            missing_data = s.recv(data_size - len(data))

            if missing_data:
                data += missing_data
            else:
                break

        memfile = BytesIO(data)
        frame = np.load(memfile, allow_pickle=True)
        print(count, type(frame))
        count += 1
            
        x = np.expand_dims(frame, axis=0)
        x = preprocess_input(x)
        print(type(x))
        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])

        #print("frame????? ", type(frame)," ", frame)
        #_, enc = cv2.imencode('.jpg', frame)
        #cv2.imshow("r", frame)
        #key = cv2.waitKey(1) & 0xFF
        #if key == ord('q'):
        #   sys.exit()
    else:
        s.close()
        break












#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#print(type(x))
#preds = model.predict(x)
#print('Predicted:', decode_predictions(preds, top=3)[0])
