import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import time


sys.path.append("/home/pi/.local/lib/python3.9/site-packages/")


MODEL_NAME = '../model/'
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

interpreter = Interpreter(model_path = PATH_TO_CKPT)
interpreter.allocate_tensors()

input_detail = interpreter.get_input_details()
output_detail = interpreter.get_output_details()

height = input_detail[0]['shape'][1]
width = input_detail[0]['shape'][2]

floating_model = (input_detail[0]['dtype']==np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_detail[0]['name']

boxes_idx, classes_idx, scores_idx = 1,3,0

video = cv2.VideoCapture(cv2.CAP_V4L2+0)
video.set(3,resW)
video.set(4,resH)
fps = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(__file__+'.avi',fourcc,fps,(resW,resH))
out2 = cv2.VideoWriter(__file__+'.avi',fourcc,fps,(resW,resH))
frame_rate_calc = 1
freq = cv2.getTickFrequency()

mask = np.zeros((480,640), dtype=np.uint8)

cnt = 0
try:
	while True:
		cnt+=1
		t1 = cv2.getTickCount()
		ret, frame1 = video.read()

		frame = frame1.copy()
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb, (width, height))
		input_data = np.expand_dims(frame_resized, axis = 0)

		blurred_img = frame1.copy()

		beforeT = time.time()
		beforelT = time.time_ns() // 1000000
		
		if floating_model:
			input_data = (np.float32(input_data)-input_mean)/input_std

		interpreter.set_tensor(input_detail[0]['index'], input_data)
		interpreter.invoke()

		boxes = interpreter.get_tensor(output_detail[boxes_idx]['index'])[0]
		classes = interpreter.get_tensor(output_detail[classes_idx]['index'])[0]
		scores = interpreter.get_tensor(output_detail[scores_idx]['index'])[0]

		for i in range(len(scores)):
			if((scores[i]>min_conf_threshold) and (scores[i]<=1.0)):
				ymin = int(max(1, (boxes[i][0]*resH)))
				xmin = int(max(1, (boxes[i][1]*resW)))
				ymax = int(min(resH, (boxes[i][2]*resH)+5))
				xmax = int(min(resW, (boxes[i][3]*resW)+5))

				for y in range(ymin,ymax):
					for x in range(xmin,xmax):
						frame[y,x] = [255,255,255]
						mask[y,x] = 255
				blurred_img = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)

				cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10,255,0), 2)

				object_name = labels[int(classes[i])]

				label = '%s: %d%%' % (object_name, int(scores[i]*100))
				labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
				label_ymin = max(ymin, labelSize[1]+10)

				cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),(xmin+labelSize[0], label_ymin+baseLine-10), (255,255,255), cv2.FILLED)
				cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0) , 2)

		cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frame_rate_calc = 1/time1

		afterT = time.time()
		cv2.putText(frame, f"{afterT-beforeT : .5f} sec",(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
		while os.path.exists('/result2/after_%03d.bmp' % cnt):
			print("after_%03d.bmp is exists" % cnt)
			os.remove('/result2/after_%03d.bmp' % cnt)
		cv2.imwrite("/home/pi/Final/22-2_EM/Dates/1217/result2/after_%03d.bmp"%cnt, frame)
		cv2.imwrite("/home/pi/Final/22-2_EM/Dates/1217/result2/blurr_%03d.bmp"%cnt, blurred_img)

		out.write(frame)
		out2.write(blurred_img)


except KeyboardInterrupt:
	print("Quit Program")
	video.release()
	out.release()



