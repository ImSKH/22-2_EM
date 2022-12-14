#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
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

	int f_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int f_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(f_w,f_h));

	printf("Open Camera\n");

	Mat img;

	int cnt = 0; int max;

	if(argc > 1){
		max = int(argv[1]);
	}
	else{
		max=50;
	}

	while(cnt<=max){
		cap.read(img);
		if(img.empty()) break;
		video.write(img);
		cnt++;
	}

	cap.release();
	video.release();
	return 0;
}

