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
	Mat img;
	if(argc>1) img = imread(argv[1], IMREAD_COLOR);
	else img = imread("Lenna.png", IMREAD_COLOR);

	if(img.empty()){
		printf("There's no Img!\n");
		return -1;
	}
	int height = img.rows;
	int width = img.cols;

	Mat gray(height, width, CV_8UC1);

	cvtColor(img, gray, COLOR_BGR2GRAY);
	imwrite("gray_lenna.bmp", gray);

	imshow("gray_lenna", gray);
	if(waitKey(30)>=0) return -1;

}