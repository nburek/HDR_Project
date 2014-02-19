#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv\cv.h>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include "rgbe.h"

using namespace cv;
using namespace std;

void printExposures(vector<float>* exp){
	for(vector<float>::iterator it = exp->begin(); it != exp->end(); ++it){
		cout<<*it<<endl;
	}
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

	FileStorage fileX("formatCheck.txt", cv::FileStorage::WRITE);
	fileX << "FormatCheck" << image;
	fileX.release();
	

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

	Mat aRough = *A;
	FileStorage fileArough("ARough.txt", cv::FileStorage::WRITE);
	fileArough << "ARough" << aRough;
	fileArough.release();

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

	Mat aSmooth = *A;
	FileStorage fileA1("A_smooth.txt", cv::FileStorage::WRITE);
	fileA1 << "A_smooth" << aSmooth;
	fileA1.release();
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

	//splits out x to a file
	FileStorage fileX("x.txt", cv::FileStorage::WRITE);
	fileX << "x" << x;
	fileX.release();

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

void printMatrixToFile(string filename, Mat m){
	/*string fileExt = filename;
	fileExt.append(".txt");
	FileStorage fs(fileExt.c_str(), cv::FileStorage::WRITE);

	cout<<"Writing file "<<filename.c_str()<<endl;
	//fs << filename << ptvec;
	for(int i = 0; i < m.size().height; i++){
		for( int j = 0; j < m.size().height; j++ )
		{
			float data = m.at<float>(i,j);
			string label = ""+i;
			fs << " " << data;
		}
	}

    fs.release();
	*/

	string fileExt = filename;
	fileExt.append(".txt");

	ofstream myfile;
	myfile.open (fileExt.c_str());

	for(int i = 0; i < m.size().height; i++){
		for( int j = 0; j < m.size().width; j++ )
		{
			float data = m.at<float>(i,j);
			myfile <<data<<",\n";
		}
	}

	myfile.close();
}

void estimateRadianceMap(){

}

void writeHDRImage(){

}

int main( int argc, char** argv )
{
	vector<Mat>* photos = new vector<Mat>();
	vector<float>* exposures = new vector<float>();
	const char* pictureFolder = "C:\\Users\\Ein\\Desktop\\CathedralTest\\pictures\\";
	const char* exposureFile = "C:\\Users\\Ein\\Desktop\\CathedralTest\\exposures\\memorial.hdr_image_list.txt";
	
	//load pictures from folder
	loadPhotos(photos, pictureFolder);
	
	//load shutter speeds from text file
	loadExposures(exposures, exposureFile);

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
	splitChannelOnMatVec(photos, redPhotos, 0);

	vector<Mat>* greenPhotos = new vector<Mat>();
	splitChannelOnMatVec(photos, greenPhotos, 1);

	vector<Mat>* bluePhotos = new vector<Mat>();
	splitChannelOnMatVec(photos, bluePhotos, 2);
	cout<<"Done"<<endl;

	//Randomly sample middle image (ideally the middle image is neither too light or too dark)
	cout<<"Sampling pixels...";
	sampleImage(samplePicRed, samples, 50);
	cout<<"Done"<<endl;
	
	calcResponseCurve(&redg, &redlogE, samplePicRed, exposures, redPhotos, samples, .99f);
	//calcResponseCurve(&greeng, &greenlogE, samplePicRed, exposures, greenPhotos, samples, .99f);
	//calcResponseCurve(&blueg, &bluelogE, samplePicRed, exposures, bluePhotos, samples, .99f);

	//write log E_n out to file
	FileStorage fileLog("log_E.txt", cv::FileStorage::WRITE);
	fileLog << "log_E" << redlogE;
	fileLog.release();

	FileStorage fileZ("log_E.txt", cv::FileStorage::WRITE);
	fileLog << "log_E" << redlogE;
	fileLog.release();

	printMatrixToFile("g", redg);

	cout<<"Program Executed Without Issue"<<endl;

	//These do nothing at the moment
	estimateRadianceMap();		//this may get absorbed into the response curve

	writeHDRImage();
    return 0;
}


