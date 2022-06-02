import cv2
import numpy as np
import socket
import struct
from io import BytesIO
import sys

cap = cv2.VideoCapture(sys.argv[1])

#client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#client_socket.connect(('192.168.122.235', 1234))
host = socket.gethostbyname(socket.gethostname())
port = 1234
server_socket = socket.create_server((host, port))

print('waiting at ', host, ' ', port)
server_socket.listen()
conn, addr = server_socket.accept()

print('connected to ', addr)
while cap.isOpened():
    _, frame = cap.read()

    memfile = BytesIO()
    np.save(memfile, frame)
    memfile.seek(0)
    data = memfile.read()

    # Send form byte array: frame size + frame content
    conn.sendall(struct.pack("L", len(data)) + data)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
