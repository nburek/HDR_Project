
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void getExpShift(const Mat *img1, const Mat *img2, int shift_bits, int shift_ret[2]);
void newExpShift(const Mat *img1, const Mat *img2, int shift_bits, int shift_ret[2]);

/**
 *	Subsample the image img by a factor of two in each dimension and put the result into a 
 *	newly allocated image img_ret.
 *
 */
void imageShrink2(const Mat *img, Mat *img_ret);

/**
*	Allocate and compute the threshold bitmap tb and the exclusion bitmap eb for the image
*	img.
*
*/
void computeBitmaps(const Mat *img, Mat *tb, Mat *eb);

/**
*	Shift a bitmap by (xo,yo) and put the result into the preallocated bitmap bm_ret, 
*	clearing exposed border areas to zero.
*
*/
void bitmapShift(const Mat *bm, int xo, int yo, Mat *bm_ret);



int main3(int argc, char** argv)
{
	Mat img1, img2, grey1, grey2;
	img1 = imread("deskPic1.jpg", IMREAD_COLOR); // Read the file
	img2 = imread("deskPic2.jpg", IMREAD_COLOR); // Read the file
	cvtColor(img1, grey1, CV_BGR2GRAY);
	cvtColor(img2, grey2, CV_BGR2GRAY);

	int shiftVal[2];
	getExpShift(&grey1, &grey2, 6, shiftVal);

	cout << "Shift amount: (" << shiftVal[0] << "," << shiftVal[1] << ")" << endl;

	namedWindow("Greyscale", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Greyscale", grey1); // Show our image inside it.

	waitKey(0); // Wait for a keystroke in the window

	return 0;
}

void newExpShift(const Mat *img1, const Mat *img2, int shift_bits, int shift_ret[2])
{
	int min_err;
	int cur_shift[2];
	Mat tb1, tb2;
	Mat eb1, eb2;
	int i, j;

	if (shift_bits > 0)
	{
		Mat sml_img1, sml_img2;
		imageShrink2(img1, &sml_img1);
		imageShrink2(img2, &sml_img2);
		newExpShift(&sml_img1, &sml_img2, shift_bits - 1, cur_shift);
		cur_shift[0] *= 2;
		cur_shift[1] *= 2;
	}
	else
	{
		cur_shift[0] = cur_shift[1] = 0;
	}

	computeBitmaps(img1, &tb1, &eb1);
	computeBitmaps(img2, &tb2, &eb2);
	min_err = img1->cols * img1->rows;


	for (i = -1; i <= 1; i++)
		for (j = -1; j <= 1; j++)
		{
			int xs = cur_shift[0] + i; //column shift
			int ys = cur_shift[1] + j; //row shift

			Mat diff_b(img1->rows, img1->cols, CV_8UC1, Scalar(0));

			int err = 0;


			for (int row = 0; row < diff_b.rows; ++row)
			{
				if ((row - ys)<0 || (row - ys)>diff_b.rows)
				{
					continue;
				}
				else
				{
					for (int col = 0; col < diff_b.cols; ++col)
					{
						int ind1 = col + row*diff_b.step;
						if ((col - xs) < 0 || (col - xs) >= diff_b.cols)
						{
							diff_b.data[ind1] = 0;
						}
						else
						{
							int ind2 = col - xs + (row - ys)*diff_b.step;
							diff_b.data[ind1] = (((tb1.data[ind1] ^ tb2.data[ind2]) & eb1.data[ind1]) & eb2.data[ind2]);
							if (diff_b.data[ind1])
								err++;
						}

					}
				}
			}
		
			if (err < min_err)
			{
				shift_ret[0] = xs;
				shift_ret[1] = ys;
				min_err = err;
			}
		}
}


void getExpShift(const Mat *img1, const Mat *img2, int shift_bits, int shift_ret[2])
{
	int min_err;
	int cur_shift[2];
	Mat tb1, tb2;
	Mat eb1, eb2;
	int i, j;

	//shrink the image and do a recursive call until we reach the smallest size
	if (shift_bits > 0)
	{
		Mat sml_img1, sml_img2;
		imageShrink2(img1, &sml_img1);
		imageShrink2(img2, &sml_img2);
		getExpShift(&sml_img1, &sml_img2, shift_bits - 1, cur_shift);
		cur_shift[0] *= 2;
		cur_shift[1] *= 2;
	}
	else //we've reached the smallest size so start doing calculations
	{
		cur_shift[0] = cur_shift[1] = 0;
	}

	computeBitmaps(img1, &tb1, &eb1);
	computeBitmaps(img2, &tb2, &eb2);
	min_err = img1->cols * img1->rows;
	

	for (i = -1; i <= 1; i++)
		for (j = -1; j <= 1; j++)
		{
			int xs = cur_shift[0] + i;
			int ys = cur_shift[1] + j;
			Mat shifted_tb2(img1->rows,img1->cols,CV_8UC1,Scalar(0));
			Mat shifted_eb2(img1->rows, img1->cols, CV_8UC1, Scalar(0));
			Mat diff_b(img1->rows, img1->cols, CV_8UC1, Scalar(0));

			int err;

			//shift the second image
			bitmapShift(&tb2, xs, ys, &shifted_tb2);
			bitmapShift(&eb2, xs, ys, &shifted_eb2);

			//create a difference map between the first and shifted second image
			bitwise_xor(tb1, shifted_tb2, diff_b);
			bitwise_and(diff_b, eb1, diff_b);
			bitwise_and(diff_b, shifted_eb2, diff_b);

			//check the error in the difference map and see if it's the smallest so far
			err = countNonZero(diff_b);
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
	resize(*img, *img_ret, Size(), 0.5, 0.5, INTER_AREA);
}

void computeBitmaps(const Mat *img, Mat *tb, Mat *eb)
{
	/*
		Perform a binary thresholding on the image, setting everything
		above the threshold to 1
	*/
	int thresh = 127;
	tb->create(img->size(), img->type());
	threshold(*img, *tb, thresh, 1, THRESH_BINARY);


	/*
		Get the exclusion bitmap by setting everything in the range +-4
		of the threshold to 0 and everything else to 1
	*/
	int var = 4;
	eb->create(img->size(), img->type());
	Mat temp(img->size(), img->type());

	//create a map of everything above the lower threshold being mapped to 1
	//subtract 1 from (thresh-var) to make it inclusive
	threshold(*img, temp, thresh - var-1, 1, THRESH_BINARY);
	//create a map of everything above the uppwer threshold being mapped to 0
	threshold(*img, *eb, thresh + var, 1, THRESH_BINARY_INV);
	//xor the two maps to get everything between the two thresholds
	bitwise_xor(*eb, temp, *eb);

}

void bitmapShift(const Mat *bm, int xo, int yo, Mat *bm_ret)
{
	bm_ret->create(bm->size(), bm->type());


	for (int row = 0; row < bm->rows; row++)
	{
		int oldRow = row - yo;
		for (int col = 0; col < bm->cols; col++)
		{
			int oldCol = col - xo;
			if (oldRow < 0 || oldRow >= bm->rows || oldCol < 0 || oldCol >= bm->cols)
				bm_ret->data[col + row*bm_ret->step] = 0;
			else
				bm_ret->data[col + row*bm_ret->step] = bm->data[oldCol + oldRow*bm->step];
		}
	}
}
