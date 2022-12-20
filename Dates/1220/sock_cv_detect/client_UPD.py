import socket
import cv2

ip = '192.168.45.179'
port = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cap = cv2.VideoCapture(cv2.CAP_V4L2+0)
size = 46080

while True :
	ret, frame = cap.read()
	d = frame.flatten()
	s = d.tobytes()

	for i in range(20):
		sock.sendto(s[i*size:(i+1)*size],(ip,port))