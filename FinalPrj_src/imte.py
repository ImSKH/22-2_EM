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

MODEL_NAME = '/home/pi/Final/codes/'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5
resW = 640
resH = 480
imW, imH = int(resW), int(resH)
use_TPU = 'store_true'


pkg = imporlib.util.find_spec('tflite_runtime')

if pkg:
	print("success")
else:
	print("failed")