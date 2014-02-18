
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
//#include <cmath>
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
	f = fopen("memorial.hdr", "rb");
	rgbe_header_info headerInfo;
	RGBE_ReadHeader(f, &image_width, &image_height, &headerInfo);
	hdrData = (float*)malloc(sizeof(float)* 3 * image_width*image_height);
	RGBE_ReadPixels_RLE(f, hdrData, image_width, image_height);

	float min = 10000.0f, max = 0.0f;
	for (int i = 0; i < (image_width*image_height * 3); ++i)
	{
		if (hdrData[i] < min)
			min = hdrData[i];
		if (hdrData[i] > max)
			max = hdrData[i];
	}


	Mat rgbImage(image_height, image_width, CV_32FC3, hdrData);
	//rgbImage /= max;
	Mat xyzImage(image_height, image_width, CV_32FC3);
	

	cvtColor(rgbImage, xyzImage, CV_RGB2XYZ);

	vector<Mat> channels;
	split(xyzImage, channels);


	Mat logMap;
	Mat dividend, divisor;

	//find the world adaptation luminence by finding the log-average
	int c = 1; //the channel to use for the luminance
	log(channels[c], logMap);
	Lwa = expf(float(mean(logMap)[0]));


	//find max luminence in the world and scale it by the Lwa
	double maxReturn;
	minMaxLoc(channels[c], NULL, &maxReturn);
	Lwmax = float(maxReturn)/Lwa;


	//scale the world luminance by the adaptation luminence
	channels[c] /= Lwa; //not sure if this should be multiply or divide

	//run the tonemapping algorithm on the luminance
	float c1 = (0.01f*Ldmax)/log10f(Lwmax + 1.0f);
	float c2 = logf(b)/logf(0.5f);
	
	dividend.create(channels[c].size(), channels[c].type());
	log(channels[c] + 1.0f, dividend);

	divisor.create(channels[c].size(), channels[c].type());
	pow((channels[c] / Lwmax), c2, divisor);
	log((2.0f + 8.0f*divisor), divisor);

	divide(dividend, divisor, channels[c]);
	channels[c] *= c1;

	minMaxLoc(channels[1], NULL, &maxReturn);

	
	//recombine the new luminance with the old X anx Z channels
	merge(channels, xyzImage);

	//change back to the BGR colorspace
	cvtColor(xyzImage, rgbImage, CV_XYZ2BGR);

	imwrite("testImg.jpg", rgbImage);

	namedWindow("ToneMap", WINDOW_NORMAL); // Create a window for display.
	imshow("ToneMap", rgbImage); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
