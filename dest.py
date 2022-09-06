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

import Batch

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


BATCH_SIZE = 1
IMG_DIM_X = 300
IMG_DIM_Y = 300

batch_data = np.zeros((BATCH_SIZE, IMG_DIM_X, IMG_DIM_Y,3))

time_b4_transmission = time.time()
while num_sent_frames < num_frames_to_send:
    proc_time = time.time()
    for i in range(BATCH_SIZE):



        num_sent_frames += 1
        #start_time = time.time()
        data = s.recv(payload_size)
        #end_time = time.time()
        #print("time before recv =", start_time)
        #print("time after recv =", end_time)
        #print("time to recv =", end_time - start_time)
        
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
            #print("process time =", time.time() - proc_time)
            #print(count, frame.shape)
            #count += 1
            

            batch_data[i] = frame
            #x = np.expand_dims(frame, axis=0)
            #print(x.shape)
            
        else:
            s.close()
            break
    x = Batch.Batch(preprocess_input(batch_data), BATCH_SIZE)
    #x = preprocess_input(batch)
    print("process time =", time.time() - proc_time) #print(type(x))
    preds = model.predict(x, verbose=1, use_multiprocessing=False, workers=1, max_queue_size=1, batch_size=BATCH_SIZE)
    #print(preds)
    #print('Predicted:', decode_predictions(preds, top=3)[0])

    #print("frame????? ", type(frame)," ", frame)
    #_, enc = cv2.imencode('.jpg', frame)
    #cv2.imshow("r", frame)
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord('q'):
    #   sys.exit()
    

print("Total transmission time =", time.time() - time_b4_transmission)









#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#print(type(x))
#preds = model.predict(x)
#print('Predicted:', decode_predictions(preds, top=3)[0])
