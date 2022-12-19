##https://awakening95.tistory.com/1
import socket
import cv2

UDP_IP = '192.168.45.16'
UDP_PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(cv2.CAP_V4L2+0)

try:
	while True:
		ret, frame = cap.read()
		d = frame.flatten()
		s = d.tobytes()

		for i in range(20):
			sock.sendto(bytes([i])+s[i*46080:(i+1)*46080], (UDP_IP, UDP_PORT))

except KeyboardInterrupt:
	cap.release()
	quit()