
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
//#include <cmath>
#include "rgbe.c"

using namespace cv;
using namespace std;

const int maxSliderVal = 100;
int sliderVal = 22.0f;
float minGamma = 0.0f;
float maxGamma = 10.0f;
Mat originalImage;
Mat finalImage;
float gamma = 2.2f;

void gammaCorrection(Mat &mat)
{
	pow(mat, (1.0f / gamma), mat);
}

void printMinMaxPerChannel(const string &mess, Mat &mat)
{
	vector<Mat> channels;
	split(mat, channels);
	for (int i = 0; i < channels.size(); ++i)
	{
		double min, max;
		minMaxLoc(channels[i], &min, &max);
		cout << mess << " ";
		cout << "min: " << min << ", max: " << max << endl;
	}
}

void on_trackbar(int, void*)
{
	Mat mat;
	finalImage.copyTo(mat);
	float val = (float)sliderVal / maxSliderVal;
	gamma = val * (maxGamma - minGamma) + minGamma;
	gammaCorrection(mat);
	printMinMaxPerChannel("Gamma corrected (bgr)", mat);
	imshow("Final ToneMapped", mat); // Show our image inside it.
}

void showFinalImage()
{
	Mat mat;
	finalImage.copyTo(mat);
	gammaCorrection(mat);
	namedWindow("Final ToneMapped", WINDOW_NORMAL); // Create a window for display.
	createTrackbar("GammaTrackbar", "Final ToneMapped", &sliderVal, maxSliderVal, on_trackbar);
	printMinMaxPerChannel("Gamma corrected (bgr)", mat);
	imshow("Final ToneMapped", mat); // Show our image inside it.
}

void showWindow(const string &winName, cv::InputArray mat)
{
	Mat temp(mat.size(), mat.type());
	mat.getMat().copyTo(temp);
	namedWindow(winName, WINDOW_NORMAL); // Create a window for display.
	imshow(winName, temp); // Show our image inside it.
}

/**
 * Normalize all three channels based on the max value found across all of them
 */
void normalize3Chan(Mat &mat)
{
	vector<Mat> channels;
	split(mat, channels);
	double realMin, realMax;
	for (int i = 0; i < channels.size(); ++i)
	{
		double min, max;
		minMaxLoc(channels[i], &min, &max);
		if (i == 0)
		{
			realMin = min;
			realMax = max;
		}
		else
		{
			if (realMin > min)
				realMin = min;
			if (realMax < max)
				realMax = max;
		}
	}
	mat = mat = (mat - realMin) / (realMax - realMin);
}

/**
 *	Normalize each channel based on the max value found in each individual channel
 */
void normalize3ChanInd(Mat &mat)
{
	vector<Mat> channels;
	split(mat, channels);
	for (int i = 0; i < channels.size(); ++i)
	{
		double min, max;
		minMaxLoc(channels[i], &min, &max);
		channels[i] = (channels[i] - min) / (max - min);
	}
	merge(channels, mat);
}

void normalize1Chan(Mat &mat)
{
	double min,max;
	minMaxLoc(mat, &min, &max);
	mat = (mat - min) / (max - min);

}

void scaleTo255(Mat &mat)
{
	mat *= 255.0f;
}


void bandPass(Mat &mat, float min, float max)
{
	vector<Mat> channels;
	split(mat, channels);

	for (int i = 0; i < channels.size(); ++i)
	{
		threshold(channels[i], channels[i], max, 0.0f, THRESH_TRUNC);
		threshold(channels[i], channels[i], min, 0.0f, THRESH_TOZERO);
	}
	merge(channels, mat);
}


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


	Mat rgbImage(image_height, image_width, CV_32FC3, hdrData);
	Mat xyzImage(image_height, image_width, CV_32FC3);

	Mat foo;
	//normalizing the input image and displaying some originals
	cvtColor(rgbImage, foo, CV_RGB2BGR);
	printMinMaxPerChannel("original(bgr)", foo);
	showWindow("OriginalBGR", foo);
	normalize3Chan(foo);
	printMinMaxPerChannel("normalized original(bgr)", foo);
	scaleTo255(foo);
	printMinMaxPerChannel("normalized original scaled 255(bgr)", foo);
	normalize3Chan(rgbImage);
	gammaCorrection(foo);
	printMinMaxPerChannel("normalized gamma corrected (bgr)", foo);
	showWindow("NormalizedBGRWithGamma", foo);

	//convert from RGB to XYZ
	cvtColor(rgbImage, xyzImage, CV_RGB2XYZ);

	//get the luminance channel from the XYZ
	vector<Mat> channels;
	printMinMaxPerChannel("original XYZ ", xyzImage);
	split(xyzImage, channels);


	Mat logMap;
	Mat dividend, divisor;

	//find the world adaptation luminence by finding the log-average
	int c = 1; //the channel to use for the luminance
	log(channels[c], logMap);
	Lwa = expf(float(mean(logMap)[0]));
	cout << "Lwa: " << Lwa << endl;


	//find max luminence in the world and scale it by the Lwa
	double maxReturn;
	minMaxLoc(channels[c], NULL, &maxReturn);
	Lwmax = float(maxReturn) / Lwa;
	cout << "Lwmax: " << Lwmax << endl;


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

	normalize1Chan(channels[c]);
	showWindow("NewLuminance", channels[c]);

	//recombine the new luminance with the old X anx Z channels
	merge(channels, xyzImage);
	printMinMaxPerChannel("new XYZ ", xyzImage);

	//normalize3ChanInd(xyzImage);

	//change back to the BGR colorspace
	cvtColor(xyzImage, rgbImage, CV_XYZ2BGR);
	printMinMaxPerChannel("tone mapped (bgr)", rgbImage);


	//normalize3Chan(rgbImage);
	//scaleTo255(rgbImage);

	rgbImage.copyTo(finalImage);
	showFinalImage();

	/*vector<Mat> newChan;
	split(rgbImage, newChan);
	showWindow("ToneChannel", newChan[2]);*/

	imwrite("testImg.jpg", rgbImage);
	//namedWindow("ToneMap", WINDOW_NORMAL); // Create a window for display.
	//imshow("ToneMap", rgbImage); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}
