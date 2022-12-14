import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

sys.path.append("/home/pi/.local/lib/python3.9/site-packages/")


class VideoStream:
    def __init__(self, resolution=(640, 480), framerate=60):
        self.stream = cv2.VideoCapture(cv2.CAP_V4L2+0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        fps = self.stream.get(cv2.CAP_PROP_FPS)
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

    def fps(self):
        return self.fps

    def stop(self):
        self.stopped = True


MODEL_NAME = '/home/pi/Final/22-2_EM/codes/'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5
resW = 640
resH = 480
imW, imH = int(resW), int(resH)
use_TPU = 'store_true'

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

if labels[0] == '???':
    del(labels[0])

interpreter = Interpreter(model_path = PATH_TO_CKPT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']

if('StatefulPartitionedCall' in outname):
    boxes_idx, classes_idx, scores_idx = 1,3,0
else:
    boxes_idx, classes_idx, scores_idx = 0,1,2

frame_rate_calc = 1
freq = cv2.getTickFrequency()

videostream = VideoStream(resolution=(imW, imH), framerate = 60).start()
time.sleep(1)

fps = videostream.fps()

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
outVideo = cv2.VideoWriter("outVideo_test.avi",fourcc,fps,(640,480))

delay = round(1000/fps)

while True:
    t1 = cv2.getTickCount()
    frame1 = videostream.read()

    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    boxing_img = frame1.copy()

    if floating_model:
        input_data = (np.float32(input_data)-input_mean)/input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    for i in range(len(scores)):
        if((scores[i]>min_conf_threshold) and (scores[i]<=1.0)):
            ymin = int(max(1, (boxes[i][0]*imH)))
            xmin = int(max(1, (boxes[i][1]*imW)))
            ymax = int(min(imH, (boxes[i][2]*imH)+5))
            xmax = int(min(imW, (boxes[i][3]*imW)+5))

            cv2.rectangle(boxing_img, (xmin,ymin), (xmax,ymax), (10,255,0), 2)

            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1]+10)
            cv2.rectangle(boxing_img, (xmin, label_ymin-labelSize[1]-10),(xmin+labelSize[0], label_ymin+baseLine-10), (255,255,255), cv2.FILLED)
            cv2.putText(boxing_img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0) , 2)
            
    cv2.putText(boxing_img, 'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    outVideo.write(boxing_img)

    if cv2.waitKey(delay) == 27:
        break

videostream.stop()
videostream.release()