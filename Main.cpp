
#include <string.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;



int main( int argc, char** argv ) {
	namedWindow("BoW");

	string imgurl;
	Mat initial_img;

	cout << "Select an image: ";
	cin >> imgurl;
	cout << endl;

	initial_img = cv::imread(imgurl, 1);
	imshow("BoW", initial_img);

	waitKey(0);

	return 0;
}