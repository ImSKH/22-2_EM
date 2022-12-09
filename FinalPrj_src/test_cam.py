import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import copy
import sys

sys.path.append("/home/pi/.local/lib/python3.9/site-packages/tflite_runtime/__init__.py")

class VideoStream:
	def __init__(self, resolution=(640, 480), framerate=60):
		self.stream = cv2.VideoCapture.open(0)
		ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
		ret = self.stream.set(3, resolution[0])
		ret = self.stream.set(4, resolution[1])
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False

	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		while True:
			if self.stopped:
				self.stream.release()
				return

			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True


videostream = VideoStream(resolution=(imW,imH), framerate = 60).start()