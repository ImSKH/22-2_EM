import cv2
import sys

cap = cv2.VideoCapture(cv2.CAP_V4L2+0)
cap.set(3,640)
cap.set(4,480)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outVideo = cv2.VideoWriter("outVideo_VideoWriteTest.avi", fourcc, 10, (640,480))

while True:
	(grabbed, frame) = cap.read()
	outVideo.write(frame)

	a = input():

	if a == "q":
		cap.release()
		sys.exit()
