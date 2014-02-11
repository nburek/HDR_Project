
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void getExpShift(const Mat *img1, const Mat *img2, int shift_bits, int shift_ret[2]);
void imageShrink2(const Mat *img, Mat *img_ret);
void computeBitmaps(const Mat *img, Mat *tb, Mat *eb);
void bitmapShift(const Mat *bm, int xo, int yo, Mat *bm_ret);
void bitmapXOR(const Mat *bm1, const Mat *bm2, Mat *bm_ret);
void bitmapAND(const Mat *bm1, const Mat *bm2, Mat *bm_ret);
int bitmapTotal(const Mat *bm);

int main(int argc, char** argv)
{
	Mat greyMat, colorMat, resizedImg;
	colorMat = imread("exampleImage.jpg", IMREAD_COLOR); // Read the file
	cvtColor(colorMat, greyMat, CV_BGR2GRAY);

	cout << colorMat.channels() << endl;
	cout << greyMat.channels() << endl;

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", resizedImg); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window
}


void getExpShift(const Mat *img1, const Mat *img2, int shift_bits, int shift_ret[2])
{
	int min_err;
	int cur_shift[2];
	Mat tb1, tb2;
	Mat eb1, eb2;
	int i, j;

	if (shift_bits > 0)
	{

	}
	else
		cur_shift[0] = cur_shift[1] = 0;

	computeBitmaps(img1, &tb1, &eb1);
	computeBitmaps(img1, &tb2, &eb2);
	min_err = img1->col * img1->rows;

	for (i = -1; i <= 1; i++)
		for (j = -1; j <= 1; j++)
		{
			int xs = cur_shift[0] + i;
			int ys = cur_shift[1] = j;
			Mat shifted_tb2(img1->rows,img1->cols,CV_8UC1);
			Mat shifted_eb2(img1->rows, img1->cols, CV_8UC1);
			Mat diff_b(img1->rows, img1->cols, CV_8UC1);
			int err;

			bitmapShift(&tb2, xs, ys, &shifted_tb2);
			bitmapShift(&eb2, xs, ys, &shifted_eb2);
			bitmapXOR(&tb1, &shifted_tb2, &diff_b);
			bitmapAND(&diff_b, &eb1, &diff_b);
			bitmapAND(&diff_b, &shifted_eb2, &diff_b);
			err = bitmapTotal(&diff_b);
			if (err < min_err)
			{
				shift_ret[0] = xs;
				shift_ret[1] = ys;
				min_err = err;
			}
		}
}


void imageShrink2(const Mat *img, Mat *img_ret)
{
	resize(*img, *img_ret, Size(), 0.5, 0.5, CV_INTER_AREA);
}

void computeBitmaps(const Mat *img, Mat *tb, Mat *eb)
{

}

void bitmapShift(const Mat *bm, int xo, int yo, Mat *bm_ret)
{

}

void bitmapXOR(const Mat *bm1, const Mat *bm2, Mat *bm_ret)
{

}

void bitmapAND(const Mat *bm1, const Mat *bm2, Mat *bm_ret)
{

}

int bitmapTotal(const Mat *bm)
{

}
