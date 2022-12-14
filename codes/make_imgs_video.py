import cv2
import glob
import numpy as np
import re

img_arr =[]
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2]= map(int, parts[1::2])
    return parts

for filename in sorted(glob.glob('/home/pi/Final/codes/result/tmp*.jpg'), key = numericalSort):
    img = cv2.imread(filename)
    h, w, l = img.shape
    size = (w, h)
    img_arr.append(img)

out = cv2.VideoWriter('/home/pi/Final/codes/outPy.mp4', cv2.VideoWriter_fourcc(*'H264'),10,size)

for i in range(len(img_arr)):
    out.write(img_arr[i])
out.release()


