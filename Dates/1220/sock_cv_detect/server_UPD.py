import socket
import numpy
import cv2

ip = '192.168.45.179'
port = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

size = 46080

s = b""

while True :
	data, addr = sock.recvfrom(size)
	s += data

	if len(s) == (size*20):
		frame = numpy.fromstring(s, dtype = numpy.uint8)
		frame = frame.reshape(480,640,3)
		cv2.imshow("frame", frame)
		s = b""

		if cv2.waitKey(1)&0xFF == ord('q'):
			cv2.destroyAllWindows()
			break