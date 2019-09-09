// 
//                           License Agreement
//							  for this project
//
// 2019 Manuel Moreno
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// for OpenCV licence check the attached files
// 

// Tests the generation and factorization of symmetric matrices with the opencv
// functions/ specific for the DLT 9x9 and 12x12 system.
// warning: only tested with visual studio 10, but would be fine in linux
// (clock.hpp changes on both systems)
// tests:
//         A. gen DLT 12x12 symmetric matrixes
//         B. gen generic nxn symmetric matrixes, and run UDU', UDV' 
//         C. gen DLT 9x9, 12x12 symmetric matrixes, and run UDU', UDV'
//         D. gen DLT 12x12 for n points, and run UDU', UDV'

// This test, comes from a personal question, why in opencv, 
//		implicit, manual A'A triangle + cvCompleteSymm + UDU'(eigenvv) is used 
//      by CvHomographyEstimator::runKernel:, a DLT 9x9
//    and 
//		explicit DLT system in A + cvMulTransposed + UDV'(SVD.compute) is used 
//      by cvFindExtrinsicCameraParams2, a DLT 12x12
//
//    when...
//      - for the symmetric matrixes a UDU' decomposition would expected to 
//        run better (as in runKernel)
//      - making the DLT A'A triangle manually in a loop would be expected to be better
//        than multiplying by the transpose of the DLT A (as in runKernel)  
//        (anyway cvMulTransposed will call cvCompleteSymm)
//         - the DLT 12x12 triangle can be created manually as in the 9x9 case (as in runKernel)  .

// USE:
//   #define USE_FP32 if you want to used floats instead of doubles
//
//   i used in a project with 2 configs were _DEBUG i set no optimization flags
//   and RELEASE full optimization; so if that is the case, it will output
//   the strings "VC10 not opt" and "VC10 OX" in the plots;
//
//   the output are 4 plots that are appended to "res.txt" that you can 
//   insert them all in "plot_test.m" to see the results (uses uiaxes.m).


#include <iostream>
#include "clock.hpp"
#include "udut_givens.hpp"
#include "udvt_givens.hpp" 

//#define USE_FP32

#ifdef USE_FP32
	#define FP float
	#define CV_FP CV_32F
	#define FP_STR "fp32"
#else
	#define FP double
	#define CV_FP CV_64F
	#define FP_STR "fp64"
#endif

#ifdef _DEBUG
	const char* configName = "VC10 not opt.";
#else
	const char* configName = "VC10 OX";
#endif 

void printUpdate()
{
	char name[500];
	sprintf(name, "%s %s",configName, FP_STR);
	std::cout << "\nrunning " << name;
	FILE *f = fopen("res.txt","a+b");
		fprintf(f, "\n\n %% - config: %s ---------------------------------------------\n\n", name);
		fprintf(f, "\n figure(); h = uiaxes([1 4], 'border', .09);");
	fclose(f);
}

int testNum = 1;
void print(int NTEST1, int NTEST2, 
		   const char* testName,
		   const char** testStrs, 
		   const char* xAxisStr,
		   std::vector<int> &points,
		   std::vector<double> *times)
{
	#define CHAR_COMA (i==points.size()-1)?' ':','

	FILE *f = fopen("res.txt","a+b");

	char name[500];
	sprintf(name,"%s/%s %s(%i sets of %i reps per %s)", testName, configName, FP_STR, NTEST1, NTEST2, xAxisStr);
	fprintf(f, "\n\n%% %s", name);

	for (int j=0; j< 2; j++)
	{
		fprintf(f,"\nteste%i = [",j);
		for (int i=0; i< points.size(); i++)
			fprintf(f,"%f%c", times[j].at(i), CHAR_COMA);
		fprintf(f,"];");
	}
	fprintf(f,"\n x = [");
	for (int i=0; i< points.size(); i++)
		fprintf(f,"%i%c", points[i], CHAR_COMA);
	fprintf(f,"];");

	fprintf(f, "\naxes(h(1,%i)); plt_('%s', x,teste0,teste1,'%s','%s','%s');\n",testNum,name,testStrs[0],testStrs[1],xAxisStr);
	fclose(f);
	testNum++;
}


//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// test the decompositions UDUt UDVt  
// avaliable in opencv, on a random symmetric matrix nxn
// optionally with a symmetric zero-block as DLT matrices 
void testFactorizationsFor(int NTEST1, int NTEST2, 
						   std::vector<double> *times,
						   int n, int zBlocks = 0)
{
	
	cv::Mat U(n, n, CV_FP), D(n, 1, CV_FP), Vt(n, n, CV_FP);

	// makes a dummy random symmetric matrix of NxN
	// using a random vector X:3xN and mult. X'X:(Nx3)(3xN)
	cv::Mat X(3, n, CV_FP);
	cv::randu(X, -1.0f, 1.0f);
	cv::mulTransposed(X, X, 1);

	// zeros a "zBlock"-sized block as the ones on the 
	// 9x9 and 12x12 DLT matrixes if zBlocks!=0 
	if (zBlocks) 
	{
		// colocar 2 blocos a 0
		X(cv::Rect(zBlocks,0,zBlocks,zBlocks)).setTo(0); //12x12: X'X(4:7,1:4) = 0
		X(cv::Rect(0,zBlocks,zBlocks,zBlocks)).setTo(0); //12x12: X'X(1:4,4:7) = 0
	}
	//otherwise, runs udut and udvt on a normal symmetric matrix
	//they are the same eigen and SVD.compute methods in opencv

	VL_DECL_CLOCK()
		
	VL_CLOCK_REP_SILTIC(NTEST1, NTEST2)
		udut(X,true,D,Vt);
	VL_CLOCK_REP_SILTOC()
	times[0].push_back(VL_CLOCK_MEAN);
		
	VL_CLOCK_REP_SILTIC(NTEST1, NTEST2)
		udvt(X,U,D,Vt,CV_SVD_MODIFY_A + CV_SVD_V_T);
	VL_CLOCK_REP_SILTOC()
	times[1].push_back(VL_CLOCK_MEAN);

}

void testFactorizations(bool special_=false)
{
	int NTEST1 = 10,
		NTEST2 = 80;
	std::vector<double> times[2]; 
	std::vector<int> points; 

	const char* testStrs[] = {"UDUt Givens","UDVt Givens"};

	if (special_)
	{
		testFactorizationsFor(NTEST1, NTEST2, times, 9, 3);	
		points.push_back(9);
		testFactorizationsFor(NTEST1, NTEST2, times, 12, 4);	
		points.push_back(12);
		print(NTEST1, NTEST2, "Symm factor. w zero blks", testStrs, "matrix size", points, times);
	}
	else
	{	

		for (int i=5; i<=80; i+=10)
		{
			testFactorizationsFor(NTEST1, NTEST2, times, i);	
			points.push_back(i);
		}
		print(NTEST1, NTEST2, "Full Symm factorization", testStrs, "matrix size", points, times);
	}
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////

// create random object points
// put them on a camera<-object frame 
// given by the se3 rotation with a 20º degree rotation on a X-axis
// and with a translation of 2,2,4 away from the camera
// and project them, as dummy input to DLT system
// so, oPoints: object points iPoints: projects
// intrisic matrix and distortion doesn't matter in this test.
#define CREATE_POINTS_AND_PROJECTIONS								\
	cv::Mat oPoints(i, 3, CV_64F);									\
	cv::randu(oPoints, -1.5f, 1.5f);								\
	cv::Mat iPoints(i, 2, CV_64F);									\
	FP tD[] = { 2.0f, 2.0f, 4.0f };									\
	FP rD[] = { VL_DEG2RAD(20), 0, 0 };								\
	cv::Mat tVec(3,1, CV_64F, tD), rVec(3,1, CV_64F, rD);			\
	std::vector<cv::Vec2f> projVec(i);								\
	cv::projectPoints(oPoints, rVec, tVec, 							\
		cv::Mat::eye(3,3,CV_64F), cv::Mat(), iPoints);				\
	 
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// generate the DLT system and generate the symmetric matrix
// by multipling with the trasponse using cvMulTransposed_ (as cvFindExtrinsicCameraParams2)
inline void testGenSymmMulTransp(double LtL[12][12], cv::Mat& opoints, cv::Mat& ipoints, int n)
{
	double* L; 
	CvMat _LL = cvMat( 12, 12, CV_64F, (double*)LtL );
	CvPoint3D64f* M = (CvPoint3D64f*)opoints.data;
	CvPoint2D64f* mn = (CvPoint2D64f*)ipoints.data;
	
    cv::Ptr<CvMat> matL = cvCreateMat( 2*n, 12, CV_64F );
	L = (double*)matL->data.db;

	for(int i = 0; i < n; i++, L += 24 )
	{
		double x = -mn[i].x, y = -mn[i].y;
		L[0] = L[16] = M[i].x;
		L[1] = L[17] = M[i].y;
		L[2] = L[18] = M[i].z;
		L[3] = L[19] = 1.;
		L[4] = L[5] = L[6] = L[7] = 0.;
		L[12] = L[13] = L[14] = L[15] = 0.;
		L[8] = x*M[i].x;
		L[9] = x*M[i].y;
		L[10] = x*M[i].z;
		L[11] = x;
		L[20] = y*M[i].x;
		L[21] = y*M[i].y;
		L[22] = y*M[i].z;
		L[23] = y;
	}

	cvMulTransposed_( matL, &_LL,  1 );
}
// generate the triangle of the DLT 12x12 symmetric matrix manually and
// call to fill the rest with cvCompleteSymm_ (as CvHomographyEstimator::runKernel:)
inline void testGenSymmManual(double LtL[12][12], cv::Mat& opoints, cv::Mat& ipoints, int n)
{
	
    CvMat _LtL = cvMat( 12, 12, CV_64F, LtL );
	 
	CvPoint3D64f* M = (CvPoint3D64f*)opoints.data;
	CvPoint2D64f* m = (CvPoint2D64f*)ipoints.data;
    cvZero( &_LtL );
    for( int i = 0; i < n; i++ )
    {
        const double Lx[] = { M[i].x, M[i].y, M[i].z, 1, 
							  0, 0, 0, 0, 
							  -(M[i].x * m[i].x), 
							  -(M[i].y * m[i].x), 
							  -(M[i].z * m[i].x), 
							  -m[i].x };
		const double Ly[] = {  0, 0, 0, 0, 
								M[i].x, M[i].y, M[i].z, 1, 
							  -(M[i].x * m[i].y), 
							  -(M[i].y * m[i].y), 
							  -(M[i].z * m[i].y), 
							  -m[i].y };					  
        int j, k;
        for( j = 0; j < 12; j++ )
            for( k = j; k < 12; k++ )
                LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
    }

    cvCompleteSymm_( &_LtL, false ); // copiar triangular superior para inferior
 }
inline void testGenSymmFor(int NTEST1, int NTEST2, 
						   std::vector<double> *times,
						   cv::Mat& opoints, cv::Mat& ipoints,
						   int n)
{
	double LtL[12][12];

	VL_DECL_CLOCK()
	
	VL_CLOCK_REP_SILTIC(NTEST1, NTEST2)
		testGenSymmManual(LtL,opoints,ipoints,n);
	VL_CLOCK_REP_SILTOC()
	times[0].push_back(VL_CLOCK_MEAN);
		
	VL_CLOCK_REP_SILTIC(NTEST1, NTEST2)
		testGenSymmMulTransp(LtL,opoints,ipoints,n);
	VL_CLOCK_REP_SILTOC()
	times[1].push_back(VL_CLOCK_MEAN);

}
void testGenSymm()
{
	int NTEST1 = 10,
		NTEST2 = 80;
	std::vector<double> times[3]; 
	std::vector<int> points; 

	const char* testStrs[] = {"DTL w cvCompleteSymm","DTL w MultTranspose"};
	
	for (int i=5; i<= 100; i+=10)
	{
		CREATE_POINTS_AND_PROJECTIONS
		testGenSymmFor(NTEST1, NTEST2, times, oPoints, iPoints, i);
		points.push_back(i);
	}
	print(NTEST1, NTEST2, "DTL symm gen", testStrs, "n. points", points, times);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// test all togheter
inline void testAllFor(int NTEST1, int NTEST2, 
						   std::vector<double> *times,
						   cv::Mat& opoints, cv::Mat& ipoints,
						   int n)
{
	double LtL[12][12];
	cv::Mat A(12,12,CV_FP,(double*)LtL);
	cv::Mat U(12,12,CV_FP),
			D(1,12,CV_FP),
			Vt(12,12,CV_FP);

	VL_DECL_CLOCK()
	
	VL_CLOCK_REP_SILTIC(NTEST1, NTEST2)
		testGenSymmManual(LtL,opoints,ipoints,n);
		udut(A, true, D, Vt);
	VL_CLOCK_REP_SILTOC()
	times[0].push_back(VL_CLOCK_MEAN);
		
	VL_CLOCK_REP_SILTIC(NTEST1, NTEST2)
		testGenSymmMulTransp(LtL,opoints,ipoints,n);
		udvt(A, U, D, Vt, CV_SVD_MODIFY_A + CV_SVD_V_T);
	VL_CLOCK_REP_SILTOC()
	times[1].push_back(VL_CLOCK_MEAN);

}
void testAll()
{
	int NTEST1 = 10,
		NTEST2 = 80;
	std::vector<double> times[2]; 
	std::vector<int> points; 

	const char* testStrs[] = {"DTL w cvCompleteSymm + UDUt","DTL w MultTranspose + UDVt"};
	
	for (int i=5; i<= 100; i+=5)
	{
		CREATE_POINTS_AND_PROJECTIONS
		testAllFor(NTEST1, NTEST2, times, oPoints, iPoints, i);
		points.push_back(i);
	}
	print(NTEST1, NTEST2, "DTL homogeneous system", testStrs, "n. points", points, times);
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
//test if UDU' and SVD' produces the same result.
//see that the decompositions can output different
//signs for the vectors, but the the solution and the 
//product of the decomposition is the same (with a very small error
//due the methods and the default stop criterion).
void DLTTestIfSameResult()
{
	cv::Mat As[3], Us[3], Ds[3];

	int i = 12; // nº of points
	CREATE_POINTS_AND_PROJECTIONS
	double LtL[12][12];
	cv::Mat A(12,12,CV_FP,(double*)LtL);
	cv::Mat U(12,12,CV_FP),
			D(1,12,CV_FP),
			Vt(12,12,CV_FP);
		
	testGenSymmManual(LtL,oPoints,iPoints,i);
		//std::cout << "\n\nL'L:\n" << A;
		As[0] = A.clone();

		udut(A, true, D, Vt);
		//std::cout << "\ndiagonal:" << D.t();
		std::cout << "\nUt:\n" << Vt;
	
		Us[0] = Vt.clone();
		Ds[0] = D.clone();
	
	testGenSymmMulTransp(LtL,oPoints,iPoints,i);
		//std::cout << "\n\nL'L:\n" << A;
		As[1] = A.clone();

		udvt(A, U, D, Vt, CV_SVD_MODIFY_A + CV_SVD_V_T);
		//std::cout << "\ndiagonal:" << D.t();
		//std::cout << "\nU:\n" << U;
		std::cout << "\nVt:\n" << Vt;

		Us[1] = Vt.clone();
		Ds[1] = D.clone();

		cv::absdiff(As[0], As[1], As[2]);
		std::cout << "\n\ndiff L'L:\n"<< As[2] << "\nERR: "<< cv::sum(As[2]).val[0];
	
		cv::absdiff(Ds[0], Ds[1], Ds[2]);
		std::cout << "\n\ndiff Diag:\n"<< Ds[2].t() << "\nERR: "<< cv::sum(Ds[2]).val[0];
		
		cv::absdiff(-Us[0], Us[1], Us[2]);
		//std::cout << "\n\ndiff V'<->U':\n"<< Us[2] << "\nERR: "<< cv::sum(Us[2]).val[0];
		
		//std::cout << "\n\n";
		//cv::Mat m = Us[0].col(11).clone(); 
}

void main()
{
	printUpdate();
	//DLTTestIfSameResult();
	testGenSymm();				// gen DLT 12x12 symmetric matrixes
	testFactorizations();		// gen generic nxn symmetric matrixes, 
							    //    and run UDU', UDV'
	testFactorizations(true);   // gen DLT 9x9, 12x12 symmetric matrixes, and run UDU', UDV'
	testAll();					// gen DLT 12x12 for n points, and run UDU', UDV'
}
