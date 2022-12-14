import cv2
import sys

cap = VideoCapture(cv2.CAP_V4L2+0)
cap.set(3, 640)
cap.set(4, 480)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('WriteTest.avi',fourcc,fps,(640,480))

delay = round(1000/fps)

while True:
	ret, frame = cap.read()

	out.write(frame)

	if cv2.waitKey(delay) == 27:
		break

cap.release()
out.release()