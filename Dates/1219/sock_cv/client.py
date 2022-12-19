import cv2
import socket
import pickle
import struct

ip = '192.168.45.179'
port = 8080

capture = cv2.VideoCapture(cv2.CAP_V4L2+0)

capture.set(3, 640)
capture.set(4, 480)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
	client_socket.connect((ip,port))

	print("Connection Successed")

	while True:
		retval, frame = capture.read()
		retval, frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

		frame = pickle.dumps(frame)

		print("Transmitted frame size : {} bytes".format(len(frame)))

		client_socket.sendall(struct.pack(">L",len(frame))+frame)

capture.release()