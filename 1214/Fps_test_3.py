import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

sys.path.append("/home/pi/.local/lib/python3.9/site-packages/")

MODEL_NAME = os.getcwd() + '/model/'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = float(0.5)
resW = 640
resH = 480


pkg = importlib.util.find_spec('tflite_runtime')

if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    print("failed")
    sys.exit()


CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

itp = Interpreter(model_path = PATH_TO_CKPT)
itp.allocate_tensors()

inde = itp.get_input_details()
oude = itp.get_output_details()
h = inde[0]['shape'][1]
w = inde[0]['shape'][2]

fl_mo = (inde[0]['dtype']==np.float32)

input_mean = 127.5
input_std = 127.5

outname = oude[0]['name']

if('StatefulPartitionedCall' in outname):
    boxes_idx, classes_idx, scores_idx = 1,3,0
else:
    boxes_idx, classes_idx, scores_idx = 0,1,2

video = cv2.VideoCapture(cv2.CAP_V4L2+0)
cap.set(3, resW)
cap.set(4, resH)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(__file__+'.avi',fourcc,fps,(resW,resH))
#out2 = cv2.VideoWriter(__file__+'2.avi',fourcc,fps,(resW,resH))

while(cap.isOpened()):
	ret, frame = cap.read()
	if not ret:
		print('Camera Error!')
		sys.exit()

	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_resized = cv2.resize(frame_rgb. (w,h))
	input_data = np.expand_dims(frame_resized, axis=0)

	if fl_mo:
		input_data = (n.float32(input_data)-input_mean)/input_std

	itp.set_tensor(inde[0]['index'],input_data)
	itp.invoke()

	boxes = itp.get_tensor(oude[boxes_idx]['index'])[0]
	classes = itp.get_tensor(oude[classes_idx]['index'])[0]
	scores = itp.get_tensor(oude[scores_idx]['index'])[0]

	for i in range(len(scores)):
		if((scores[i]>min_conf_threshold) and (scores[i]<=1.0)):
			ymin = int(max(1,(boxes[i][0] * resH)))
			xmin = int(max(1,(boxes[i][1] * resW)))
			ymax = int(min(imH,(boxes[i][2] * resH)))
			xmax = int(min(imW,(boxes[i][3] * resW)))

			cv2.rectangle(frame, (xmin, ymin), (xmax,ymax), (10,255,0), 4)

			object_name = labels[int(classes[i])]
			label = '%s: %d%%' % (object_name, int(scores[i]*100))
			labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
			label_ymin = max(ymin, labelSize[1] + 10)
			cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
			cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

	out.write(frame)