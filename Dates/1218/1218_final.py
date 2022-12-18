import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import time
import I2C_LCD_driver
import RPi.GPIO as GPIO

#tflite_runtime pkg import
sys.path.append("/home/pi/.local/lib/python3.9/site-packages/")
from tflite_runtime.interpreter import Interpreter

################ Initializing ###################
#LCD initializing
lcd = I2C_LCD_driver.lcd()
lcd.backlight(1)

#Button initializing
GPIO.setmode(GPIO.BCM)
BUTT = 17
GPIO.setup(BUTT, GPIO.IN)
state = 0

#Wave Sensor initializing
TRIG, ECHO = 23,24
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)
pre_dist = 0

#TFLite Model initializing
MODEL_NAME = '../model/'
GRAPH_NAME = 'detect.tflite'
LABEL_NAME = 'labelmap.txt'
min_conf_threshold = float(0.5)
resW, resH = 640, 480
PATH_TO_CKPT = os.path.join(os.getcwd(), MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(os.getcwd(), MODEL_NAME, LABEL_NAME)

with open(PATH_TO_LABELS, 'r') as f:
	labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path = PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean, input_std = 127.5, 127.5
outname = output_details[0]['name']
boxes_idx, classes_idx, scores_idx = 1,3,0

#VideoCapture initializing
video = cv2.VideoCapture(cv2.CAP_V4L2+0)
video.set(3,resW)
video.set(4,resH)
filename = __file__.split('.')[0]
out = cv2.VideoWriter(filename+'.avi',cv2.VideoWriter_fourcc(*'DIVX'),7,(resW,resH))
out2 = cv2.VideoWriter(filename+'2.avi',cv2.VideoWriter_fourcc(*'DIVX'),7,(resW,resH))
################ Initializing END ###################


################ Function Define ##################
##WaveSensor func.
def WaveSensor():
	global pre_dist
	GPIO.output(TRIG,True)
	time.sleep(0.00001) ##pulse 1us
	GPIO.output(TRIG,False)
	while GPIO.input(ECHO) == 0:
		start = time.time()
	while GPIO.input(ECHO) == 1:
		stop = time.time()
	ctime = stop-start
	dist = ctime*34300/2
	if dist>1022:
		dist = pre_dist
	pre_dist = dist 
	return dist

##Detection and VideoWrite
def BrandDetect():
	ret, frame1 = video.read()
	frame = frame1.copy()
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame_resized = cv2.resize(frame_rgb, (width, height))
	input_data = np.expand_dims(frame_resized, axis = 0)
	mask = np.zeros((resH, resW), dtype=np.uint8)
	blurred_img = frame1.copy()
	input_data = (np.float32(input_data)-input_mean)/input_std
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()

	boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
	classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
	scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

	for i in range(len(scores)):
		if((scores[i]>min_conf_threshold) and (scores[i]<=1.0)):
			ymin = int(max(1, (boxes[i][0]*resH)))
			xmin = int(max(1, (boxes[i][1]*resW)))
			ymax = int(min(resH, (boxes[i][2]*resH)+5))
			xmax = int(min(resW, (boxes[i][3]*resW)+5))

			for y in range(ymin, ymax):
				for x in range(xmin, xmax):
					mask[y,x] = 255
			blurred_img = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)
			cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10,255,0), 2)
			object_name = labels[int(classes[i])]

			label = '%06s: %d%%' % (object_name, int(scores[i]*100))
			labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
			label_ymin = max(ymin, labelSize[1]+10)

			cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10),(xmin+labelSize[0], label_ymin+baseLine-10), (255,255,255), cv2.FILLED)
			cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0) , 2)
			lcd.lcd_display_string(label,2)

	out.write(frame)
	out2.write(blurred_img)
################ Function Define END ###################

try:
	while True:
		inputIO = GPIO.input(BUTT)
		if inputIO == False:
			if state == 0:
				lcd.lcd_clear()
				lcd.lcd_display_string("Press Button!",1)
				time.sleep(0.3)
			else :
				now_dist = WaveSensor()
				if(now_dist < 50):
					lcd.lcd_display_string("VideoCapturing...",1)
					BrandDetect()
				else :
					lcd.lcd_display_string("Out of Range",1)
		else :
			time.sleep(0.5)
			state = state ^ 1

except KeyboardInterrupt:
	print("Quit Program!")	
	lcd.lcd_clear()
	lcd.backlight(0)
	video.release()
	out.release()
	out2.release()
	GPIO.cleanup()
