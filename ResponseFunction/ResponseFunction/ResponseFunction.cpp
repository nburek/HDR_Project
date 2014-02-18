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

void loadMat(vector<Mat>* pictures, const char* pictureFolder)
{
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (pictureFolder)) != NULL) {
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

void precalculatePixelWeights(vector<float>& weights, float zMin, float zMax){
   float thresh = 0.5*( zMin + zMax );

   for( int i = 0; i < weights.size(); i++) {
        float z = (float)i;
        if( z <= thresh ) weights[i] =  z - zMin;
        if( z  > thresh ) weights[i] =  zMax - z;
   }
}

void writeMatToFile(cv::Mat& m, const char* filename)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<"\t";
        }
        fout<<endl;
    }

    fout.close();
}

Mat formatMat(Mat image, int color){
	Mat singleChan;
	Mat result;

	/*FileStorage fileX("formatCheck.txt", cv::FileStorage::WRITE);
	fileX << "FormatCheck" << m;
	fileX.release();
	*/

	if(image.channels() > 1){
		vector<Mat> chanVec;
		split(image, chanVec);
		singleChan = chanVec[color];
	}else{
		singleChan = image;
	}

	if(singleChan.type() == CV_8UC1){
		singleChan.convertTo(result, CV_32FC1, 1/255.0);
	}else{
		result = singleChan;
	}
		
	return result;
}

void generateMatrices(Mat* A, Mat* b, vector<float>* exposures, vector<Mat>* photos, map<float, Vec2i>* samples, float lambda)
{
	float z_min = 0;
	float z_max = 255;

	int k = 0;


	//Note: samples are in random order - verify that this isn't an issue
	//Verify that you are setting the correct positions in A
	cout<<"MatTest1"<<endl;
	for( map<float, Vec2i>::iterator posIndex = samples->begin(); posIndex != samples->end(); ++posIndex) {
		Vec2i currPos = (*posIndex).second;
		int i = 0;
         for( int j = 0; j < photos->size(); j++) {
			 float zij =  (*photos)[j].at<float>(currPos);
             float wij =  weightingFunction(z_min, z_max, zij);		//wij = w(Z(i,j)+1);
              A->at<float>(k, zij) = wij;							//A(k,Z(i,j)+1) = wij; 
              A->at<float>(k, 256+i) = -wij;						//A(k,n+i)= -wij

			  //fine
              b->at<float>(k, 0) = wij*log( (*exposures)[j] );
              k++;
         }
		 i++;
    }
	//namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", *A );  
	//waitKey(0);  

	cout<<"MatTest2"<<endl;
	//verify
	//A->at<float>(k,samples->size()*photos->size()) = 1;	
	A->at<float>(k, 127) = 1;	
	k++;
	cout<<"MatTest3"<<endl;
	for(int i = 1; i < 255; i++){
		// Include the smoothness equations
		A->at<float>(k,i-1) = lambda*weightingFunction(i, z_min, z_max);				//A(k,i)=l*w(i+1); 
		A->at<float>(k,i) = -2.0f*lambda*weightingFunction(i, z_min, z_max);		//A(k,i+1)=-2*l*w(i+1);
		A->at<float>(k,i+1) = lambda*weightingFunction(i, z_min, z_max);			//A(k,i+2)=l*w(i+1);
		k++;																		//k=k+1;
	}
	cout<<"Mat Executed Without Issue"<<endl;
	
}

void calcResponseCurve(Mat* g, Mat* logE, Mat img, vector<float>* exposures, vector<Mat>* ranges, map<float, Vec2i>* samples, float lambda){
	
	int n = samples->size();
	int p = ranges->size();
	int rowsA = n*p+257;										
	int colsA = 256+n;

	cout<<"CalcRespCurveTest1"<<endl;
	Mat A;
	Mat b;
	A = Mat::zeros(rowsA, colsA, CV_32F);
	b = Mat::zeros(rowsA,1, CV_32F);	

	cout<<"n: "<<n<<endl;
	cout<<"p: "<<p<<endl;
	cout<<"A: "<<A.size().height<<"x"<<A.size().width<<endl;
	cout<<"b: "<<b.size().height<<"x"<<b.size().width<<endl;

	generateMatrices(&A, &b, exposures, ranges, samples, lambda);
	cout<<"CalcRespCurveTest4: About to solve system"<<endl;
	
	//estimate response time																
	Mat x;
	bool result = solve(A,b,x, DECOMP_SVD);			//Solve the system using SVD

	FileStorage fileX("x.txt", cv::FileStorage::WRITE);
	fileX << "x" << x;
	fileX.release();

	cout<<"x: "<<x.size().height<<"x"<<x.size().width<<endl;
	cout<<"CalcRespCurveTest5: System solve successful"<<endl;
	*g = Mat(x, Range(0,256), Range::all());					
	*logE = Mat(x, Range(256, x.size().height), Range::all());		
	cout<<"CalcRespCurveTest6: Response Curve Calculation Executed without issue"<<endl;
}

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
		//choose pixel
		int randRow = rand() % image.size().height;
		int randCol = rand() % image.size().width;

		float val = image.at<float>(randRow, randCol);

		if(samples->find(val) == samples->end()){
			Vec2i pos(randRow, randCol);
			samples->insert(pair<float, Vec2i>(val, pos));
			ctr++;
		}		
	}
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
	loadMat(photos, pictureFolder);
	
	//load shutter speeds from text file
	loadExposures(exposures, exposureFile);

	//verifies that there is an equal number of loaded images and loaded exposures
	loadCheck(exposures, photos);
	
	map<float, Vec2i>* samples = new map<float, Vec2i>();
	int samplePicIndex = photos->size()/2;
	
	Mat redlogE;
	Mat redg;
	Mat samplePicRed = formatMat((*photos)[samplePicIndex], 0); //verify colorspace

	FileStorage fileRed("redChan.txt", cv::FileStorage::WRITE);
	fileRed << "redChan" << samplePicRed;
	fileRed.release();

	vector<Mat>* redRanges = new vector<Mat>();
	for(vector<Mat>::iterator splitIt = photos->begin(); splitIt != photos->end(); ++splitIt){
		Mat redChan = formatMat(*splitIt, 0);
		redRanges->push_back(redChan);
	}
	cout<<"MainTest1"<<endl;
	sampleImage(samplePicRed, samples, 50);
	cout<<"MainTest2"<<endl;
	calcResponseCurve(&redg, &redlogE, samplePicRed, exposures, redRanges, samples, .99f);

	cout<<"MainTest3 - Dimension check for g: "<<redg.size().height<<"x"<<redg.size().width<<endl;
	cout<<"MainTest4 - Dimension check for lnE: "<<redlogE.size().height<<"x"<<redlogE.size().width<<endl;

	//write log E_n out to file
	FileStorage fileLog("log_E.txt", cv::FileStorage::WRITE);
	fileLog << "log_E" << redlogE;
	fileLog.release();

	FileStorage fileG("g.txt", cv::FileStorage::WRITE);
	fileG << "g" << redg;
	fileG.release();


	cout<<"Program Executed Without Issue"<<endl;
	estimateRadianceMap();



	writeHDRImage();
    return 0;
}


