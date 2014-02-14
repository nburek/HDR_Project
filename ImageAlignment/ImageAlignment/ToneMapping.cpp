
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
	float b = 0.85f; //bias parameter
	float Ldmax = 100.0f; //the max display luminance
	float Lwa; //the world adjustment luminance
	float Lwmax; //the world max luminance
	
	FILE *f;
	int image_width, image_height;
	float *hdrData;
	f = fopen("smallOffice.hdr", "rb");
	rgbe_header_info headerInfo;
	RGBE_ReadHeader(f, &image_width, &image_height, &headerInfo);
	hdrData = (float*)malloc(sizeof(float)* 3 * image_width*image_height);
	RGBE_ReadPixels_RLE(f, hdrData, image_width, image_height);


	Mat image(image_height, image_width, CV_32FC3, hdrData);
	cvtColor(image, image, CV_RGB2XYZ);
	vector<Mat> channels;
	split(image,channels);
	
	Mat temp, temp2;
	log(channels[1],temp);
	Lwa = mean(temp);
	minMaxLoc(channels[1],NULL,&Lwmax,NULL,NULL);
	Lwmax /= Lwa;
	float c1 = (0.01f*Ldmax)/log(Lwmax + 1.0f);
	float c2 = log(b)/log(0.5f);
	
	channels[1] /= Lwa;
	temp2.create(channels[1].size(), channels[1].type());
	temp2 = log(2.0f + 8.0f * pow((channels[1]/Lwmax),c2));
	temp.create(channels[1].size(),channels[1].type());
	temp = log(channels[1] + 1.0f);
	divide(temp,temp2,channels[1]);
	channels[1] *= c1;
	
	merge(channels,3,image);
	
	cvtColor(image,image, CV_XYZ2BGR);

	namedWindow("ToneMap", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("ToneMap", image); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
