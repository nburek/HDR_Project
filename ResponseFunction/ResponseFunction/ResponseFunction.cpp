#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv\cv.h>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include "rgbe.c"

using namespace cv;
using namespace std;

void printExposures(vector<float>* exp){
	for(vector<float>::iterator it = exp->begin(); it != exp->end(); ++it){
		cout<<*it<<endl;
	}
}

void printMatrixToFile(string filename, Mat m){
	string fileExt = filename;
	fileExt.append(".txt");

	ofstream myfile;
	myfile.open (fileExt.c_str());

	for(int i = 0; i < m.size().height; i++){
		for( int j = 0; j < m.size().width; j++ )
		{
			float data = m.at<float>(i,j);
			myfile <<data<<"\n";
		}
	}

	myfile.close();
}

//verifies that the number of pictures and the number of exposures loaded checks out
void loadCheck(vector<float>* exp, vector<Mat>* pictures){
	if(exp->size() != pictures->size()){
		cout<<"Exposure count and image count don't match:"<<endl;
		if(exp->size() > pictures->size()){
			cout<<"\tfewer pictures than exposures"<<endl;
		}else if(exp->size() < pictures->size()){
			cout<<"\tmore pictures than exposures"<<endl;
		}
	}
}

void loadPhotos(vector<Mat>* pictures, const char* pictureFolder)
{
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (pictureFolder)) != NULL) {
		cout<<"Loading images..."<<endl;
		/* print all the files and directories within directory */
		while ((ent = readdir (dir)) != NULL) {
			//ignore directories above
			if(!(strcmp(ent->d_name, "..") == 0 || strcmp(ent->d_name,".") == 0)){
				string filename = pictureFolder;
				filename.append( ent->d_name);
				Mat img = imread( filename.c_str() );
				pictures->push_back(img);
				printf ("%s\n", ent->d_name);
			}
		}
		closedir (dir);
		cout<<endl;
	} else {
		/* could not open directory */
		perror ("Count not open directory");
		//return EXIT_FAILURE;
	}
}

void loadExposures(vector<float>* exp, const char* expFile){
	ifstream infile(expFile);
	string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line);
		float exposure;
		if (!(iss >> exposure)) { 
			break; // error
		} else{
			exp->push_back(exposure);
		}

	}
}

//used for generating A and b for Ax = b
float weightingFunction(float z, float z_min, float z_max){
	float avg_z = (z_min + z_max)/2;	//will an int work?
	float result;
	if(z > avg_z){
		result = z_max-z;
	}else{
		result = z-z_min;
	}
	return result;
}

//not used 
void precalculatePixelWeights(vector<float>& weights, float zMin, float zMax){
   float thresh = 0.5f*( zMin + zMax );

   for( int i = 0; i < weights.size(); i++) {
        float z = (float)i;
        if( z <= thresh ) weights[i] =  z - zMin;
        if( z  > thresh ) weights[i] =  zMax - z;
   }
}

//changes 8U to 32F and performs any desired channel splitting
Mat formatMat(Mat image, int color){
	Mat singleChan;
	Mat result;

	if(image.channels() > 1){
		vector<Mat> chanVec;
		split(image, chanVec);
		singleChan = chanVec[color];
	}else{
		singleChan = image;
	}

	if(singleChan.type() == CV_8UC1){
		singleChan.convertTo(result, CV_32FC1);
	}else{
		result = singleChan;
	}
		
	return result;
}

void splitChannelOnMatVec(vector<Mat>* photos, vector<Mat>* dst, int color)
{
	for(vector<Mat>::iterator splitIt = photos->begin(); splitIt != photos->end(); ++splitIt){
		Mat chan = formatMat(*splitIt, color);
		dst->push_back(chan);
	}
}

//creates A and b for Ax = b
void generateMatrices(Mat* A, Mat* b, vector<float>* exposures, vector<Mat>* photos, map<float, Vec2i>* samples, float lambda)
{
	cout<<"Generating matrices...";
	float z_min = 0;
	float z_max = 255;

	int k = 0;

	//Note: samples are in random order - verify that this isn't an issue
	int i = 0;
	for( map<float, Vec2i>::iterator posIndex = samples->begin(); posIndex != samples->end(); ++posIndex) {
		Vec2i currPos = (*posIndex).second;
         for( int j = 0; j < photos->size(); j++) {
			 float zij =  (*photos)[j].at<float>(currPos);
             float wij =  weightingFunction(zij+1.0f, z_min, z_max);		//wij = w(Z(i,j)+1);
              A->at<float>(k, zij+1.0f) = wij;							//A(k,Z(i,j)+1) = wij; 
              A->at<float>(k, 256+i) = -wij;						//A(k,n+i)= -wij

			  //fine
              b->at<float>(k, 0) = wij*log( (*exposures)[j] );
              k++;
         }
		 i++;
    }

	printMatrixToFile("A_rough", *A);

	//A->at<float>(k,samples->size()*photos->size()) = 1;	
	A->at<float>(k, 129) = 1.0f;	
	k++;
	//smoothness equations
	for(int i = 1; i < 255; i++){
		A->at<float>(k,i-1) = lambda*weightingFunction((float)i, z_min, z_max);				//A(k,i)=l*w(i+1); 
		A->at<float>(k,i) = -2.0f*lambda*weightingFunction((float)i, z_min, z_max);		//A(k,i+1)=-2*l*w(i+1);
		A->at<float>(k,i+1) = lambda*weightingFunction((float)i, z_min, z_max);			//A(k,i+2)=l*w(i+1);
		k++;																		//k=k+1;
	}

	printMatrixToFile("A_smooth", *A);
	cout<<"Done"<<endl;;
	
}

void calcResponseCurve(Mat* g, Mat* logE, Mat img, vector<float>* exposures, vector<Mat>* photos, map<float, Vec2i>* samples, float lambda){
	int n = samples->size();
	int p = photos->size();
	int rowsA = n*p+255;										
	int colsA = 256+n;

	Mat A;
	Mat b;
	A = Mat::zeros(rowsA, colsA, CV_32F);
	b = Mat::zeros(rowsA,1, CV_32F);	

	generateMatrices(&A, &b, exposures, photos, samples, lambda);
	cout<<"Solving linear System...";
	
	//estimate response time																
	Mat x;
	bool result = solve(A,b,x, DECOMP_SVD);			//Solve the system using SVD
	cout<<"Success!"<<endl;

	//spits out x to a file
	printMatrixToFile("x", x);

	*g = Mat(x, Range(0,256), Range::all());					
	*logE = Mat(x, Range(256, x.size().height), Range::all());		
}

//chooses sample pixels
void sampleImage(Mat image, map<float, Vec2i>* samples, int sampleSize){
	//Question: does order matter:?
	int ctr = 0;

	int totalPixels = image.size().height*image.size().width;
	if(totalPixels < sampleSize){
		cout<<"Image contains less pixels than desired sample amount"<<endl;
		return;
	}

	if(sampleSize > 50){
		cout<<"Unecessarily big sample size"<<endl;
		return;
	}


	while(ctr < sampleSize){
		//choose pixel at random
		int randRow = rand() % image.size().height;
		int randCol = rand() % image.size().width;

		float val = image.at<float>(randRow, randCol);

		//verify that its value does not exist already in the map
		if(samples->find(val) == samples->end()){
			Vec2i pos(randRow, randCol);
			samples->insert(pair<float, Vec2i>(val, pos));
			ctr++;
		}		
	}
}

void fancyRadianceMap(Mat &redG, Mat &greenG, Mat &blueG, vector<Mat>* photos, vector<float>* exposures, float *output)
{
	int rows = (*photos)[0].rows;
	int cols = (*photos)[0].cols;

	/*Mat red, green, blue;
	exp(redG, red);
	exp(greenG, green);
	exp(blueG, blue);*/

	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			float r = 0.0f, g = 0.0f, b = 0.0f;
			float rDen = 0.0f, gDen = 0.0f, bDen = 0.0f;
			for (int i = 0; i < photos->size(); ++i)
			{
				float dT = logf((*exposures)[i]);
				Vec3b pixel = (*photos)[i].at<Vec3b>(row, col);

				r += weightingFunction(pixel[2] + 1.0f, 0.0f, 255.0f)*(redG.at<float>(pixel[2], 0) - dT);
				rDen += weightingFunction(pixel[2] + 1.0f, 0.0f, 255.0f);

				g += weightingFunction(pixel[1] + 1.0f, 0.0f, 255.0f)*(greenG.at<float>(pixel[1], 0) - dT);
				gDen += weightingFunction(pixel[1] + 1.0f, 0.0f, 255.0f);

				b += weightingFunction(pixel[0] + 1.0f, 0.0f, 255.0f)*(blueG.at<float>(pixel[0], 0) - dT);
				bDen += weightingFunction(pixel[0] + 1.0f, 0.0f, 255.0f);
			}

			output[3 * (row*cols + col)] = expf(r / rDen);
			output[3 * (row*cols + col)+1] = expf(g / gDen);
			output[3 * (row*cols + col)+2] = expf(b / bDen);
		}
	}
}


void estimateRadianceMap(Mat &redG, Mat &greenG, Mat &blueG, const Mat &photo, float *output)
{
	Mat red, green, blue;
	exp(redG, red);
	exp(greenG, green);
	exp(blueG, blue);

	for (int row = 0; row < photo.rows; ++row)
	{
		for (int col = 0; col < photo.cols; ++col)
		{
			Vec3b pixel = photo.at<Vec3b>(row, col);
			output[3 * (row*photo.cols + col)] = red.at<float>(pixel[2], 0); //red
			output[3 * (row*photo.cols + col) + 1] = green.at<float>(pixel[1], 0); //green 
			output[3 * (row*photo.cols + col) + 2] = blue.at<float>(pixel[0], 0); //blue
		}
	}
}

void writeHDRImage(int rows, int cols, float* hdrData){
	FILE *file = fopen("output.hdr", "wb");
	RGBE_WriteHeader(file, cols, rows, NULL);
	RGBE_WritePixels_RLE(file, hdrData, cols,rows);
	fclose(file);
}

int main( int argc, char** argv )
{
	vector<Mat>* photos = new vector<Mat>();
	vector<float>* exposures = new vector<float>();

	string fileRoot = "C:\\Users\\Nick\\Desktop\\HDR_Project\\ResponseFunction\\ResponseFunction\\Shelter01";
	string pictureFolder = fileRoot+"\\alignedPictures\\";
	string exposureFile = fileRoot+"\\exposures\\exp.txt.txt";
	
	//load pictures from folder
	loadPhotos(photos, pictureFolder.c_str());
	
	//load shutter speeds from text file
	loadExposures(exposures, exposureFile.c_str());

	//verifies that there is an equal number of loaded images and loaded exposures
	loadCheck(exposures, photos);
	
	map<float, Vec2i>* samples = new map<float, Vec2i>();
	int samplePicIndex = photos->size()/2;
	

	Mat redlogE;
	Mat redg;

	Mat greenlogE;
	Mat greeng;

	Mat bluelogE;
	Mat blueg;


	Mat samplePicRed = formatMat((*photos)[samplePicIndex], 0); //verify colorspace: RGB vs BGR

	cout<<"Splitting Channels...";
	//Split channels from all the existing photos and store them in their own vectors
	vector<Mat>* redPhotos = new vector<Mat>();
	splitChannelOnMatVec(photos, redPhotos, 2);

	vector<Mat>* greenPhotos = new vector<Mat>();
	splitChannelOnMatVec(photos, greenPhotos, 1);

	vector<Mat>* bluePhotos = new vector<Mat>();
	splitChannelOnMatVec(photos, bluePhotos, 0);
	cout<<"Done"<<endl;

	//Randomly sample middle image (ideally the middle image is neither too light or too dark)
	cout<<"Sampling pixels...";
	sampleImage(samplePicRed, samples, 50);
	cout<<"Done"<<endl;
	

	calcResponseCurve(&redg, &redlogE, samplePicRed, exposures, redPhotos, samples, 7.0f);
	calcResponseCurve(&greeng, &greenlogE, samplePicRed, exposures, greenPhotos, samples, 7.0f);
	calcResponseCurve(&blueg, &bluelogE, samplePicRed, exposures, bluePhotos, samples, 7.0f);

	//write log E_n out to file
	cout<<"Writing response functions to file...";
	printMatrixToFile("log_E", redlogE);
	printMatrixToFile("redG", redg);
	printMatrixToFile("greenG", greeng);
	printMatrixToFile("blueG", blueg);
	cout<<"Done"<<endl;

	cout<<"Program Executed Without Issue"<<endl;

	//These do nothing at the moment
	float *hdrData = (float*)malloc(sizeof(float)* 3 * ((*photos)[4].rows)*((*photos)[4].cols));
	//estimateRadianceMap(redg,greeng,blueg,(*photos)[4],hdrData);		//this may get absorbed into the response curve
	fancyRadianceMap(redg, greeng, blueg, photos, exposures, hdrData);

	writeHDRImage(((*photos)[4]).rows, ((*photos)[4]).cols, hdrData);
    return 0;
}


