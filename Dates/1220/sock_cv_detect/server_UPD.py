import socket
import numpy
import cv2

ip = '192.168.45.179'
port = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

size = 46080

s=[b"\xff"*size for x in range(20)]


while True :
	picture = b""
	data, addr = sock.recvfrom(size+1)
	s[data[0]] = data[1:size+1]

	if data[0] == 19:
		for i in range(20):
			picture += s[i]

		frame = numpy.fromstring(picture, dtype = numpy.uint8)
		frame = frame.reshape(480,640,3)
		cv2.imshow("frame", frame)

		if cv2.waitKey(1)&0xFF == ord('q'):
			cv2.destroyAllWindows()
			break