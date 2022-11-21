#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/photo.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	VideoCapture cap;
	cap.open("/dev/video0", CAP_V4L2);
	if(!cap.isOpened()){
		printf("Can't open Camera\n");
		return -1;
	}

	int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);

	VideoWrite video("sobel.avi", VideoWriter::fourcc('M','J','P','G'),10,Size(frame_width,frame_height));
	printf("open Camera\n");

	Mat img;
	Mat gray;
	Mat sobelX;
	Mat sobelY;
	Mat sobel;

	int cnt = 0;
	int max;

	if(argc>1)	max = int(argv[1]);
	else	max = 50;

	while(cnt <= max){
		cap.read(img);
		if(img.empty()) break;
		cvtColor(img, gray, COLOR_BGR2GRAY);
		Sobel(gray, sobelX, CV_8U, 1, 0);
		Sobel(gray, sobelY, CV_8U, 0, 1);
		sobel = abs(sobelX) + abs(sobelY);
		video.write(sobel);
		cnt++;
	}

	cap.release();
	video.release();
	return 0;
}