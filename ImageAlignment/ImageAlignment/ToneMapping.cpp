
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include "rgbe.c"

using namespace cv;
using namespace std;



int main(int argc, char** argv)
{
	FILE *f;
	int image_width, image_height;
	float *hdrData;
	f = fopen("smallOffice.hdr", "rb");
	rgbe_header_info headerInfo;
	RGBE_ReadHeader(f, &image_width, &image_height, &headerInfo);
	hdrData = (float*)malloc(sizeof(float)* 3 * image_width*image_height);
	RGBE_ReadPixels_RLE(f, hdrData, image_width, image_height);


	Mat image(image_height, image_width, CV_32FC3, hdrData);
	cvtColor(image, image, CV_RGB2BGR);


	namedWindow("ToneMap", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("ToneMap", image); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
