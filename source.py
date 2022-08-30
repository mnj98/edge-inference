import cv2
import numpy as np
import socket
import struct
from io import BytesIO
import sys

cap = cv2.VideoCapture(sys.argv[1])

#client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client_socket.connect(('192.168.122.235', 1234))
hostname = socket.gethostname()
print('hostname: ', hostname)

host = socket.gethostbyname_ex(socket.gethostname())[2][1]
#host = socket.gethostbyname(hostname)
print('host: ', host)
port = 1234
server_socket = socket.create_server((host, port))
print("server socket ", server_socket)
print('waiting at ', host, ' ', port)
server_socket.listen()
conn, addr = server_socket.accept()

num_sent_frames = 0
num_frames_to_send = int(sys.argv[2])
print('connected to ', addr)
while cap.isOpened() and num_sent_frames < num_frames_to_send:
    num_sent_frames += 1
    _, frame = cap.read()

    memfile = BytesIO()
    np.save(memfile, frame)
    memfile.seek(0)
    data = memfile.read()

    # Send form byte array: frame size + frame content
    conn.sendall(struct.pack("L", len(data)) + data)

#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
conn.shutdown(socket.SHUT_RDWR)
conn.close()
cap.release()
