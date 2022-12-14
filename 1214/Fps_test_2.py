import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

sys.path.append("/home/pi/.local/lib/python3.9/site-packages/")

cap = cv2.VideoCapture(cv2.CAP_V4L2+0)
cap.set(3, 640)
cap.set(4, 480)
fps = cap.get(cv2.CAP_PROP_FPS)

filename = os.__file__
print(filename)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(filename+'.avi',fourcc,fps,(640,480))

while True:
	ret, frame = cap.read()

	out.write(frame)

cap.release()
out.release()