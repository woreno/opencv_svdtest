// 2019 Manuel Moreno
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
// for OpenCV licence check the attached files
Direct Linear Transform (DLT) is method to recover the rotation and translation of object vectors relative to a projector where they had been projected. With both object vectors and the projection vectors on the projection plane, through an cross product relation between both, is possible to reach a homogeneous linear system Ax = 0, where x are the elements that form the transformation matrix. DLT is used to find R|T and homography (an R|T where the vectors object are fixed on a plane) matrices. In the first case, you get a matrix A with 2Nx12 elements, in the second case, a matrix A with 2Nx9 elements, both with a wellknow format. Moreover, this Ax = 0 can be solved in the least square sense, by finding a solution for the null space of A'A instead of A, requiring the A'A (9x9 or 12x12) product and followed by a UDU' decomposition to get a solution. However doing that product isn't necessary programatly (you had to generate the full A matrix that is, 2Nx9 or 2Nx12 before the transpose product). Once you know the format of A'A, it is possible to generate the upper triangle of the 9x9 or 12x12 A'A, and copying those elements the lower triangle.

Here, i test the generation of A'A and factorization of symmetric matrices as in opencv.
This test, comes from a personal question, why in opencv,

implicit, manual A'A triangle + cvCompleteSymm + UDU'(eigenvv) is used 
 by CvHomographyEstimator::runKernel:, a DLT 9x9
 
and

explicit DLT system in A + cvMulTransposed + UDV'(SVD.compute) is used 
 by cvFindExtrinsicCameraParams2, a DLT 12x12
 
when...

 - for the symmetric matrixes a UDU' decomposition would expected to 
   run better (as in runKernel)
   
 - making the DLT A'A triangle manually in a loop would be expected to be better
   than multiplying by the transpose of the DLT A (as in runKernel)  
   (anyway cvMulTransposed will call cvCompleteSymm)
   
    - the DLT 12x12 triangle can be created manually as in the 9x9 case (as in runKernel)
    
tests:

    A. gen DLT 12x12 symmetric matrixes
	
    B. gen generic nxn symmetric matrixes, and run UDU', UDV' 
	
    C. gen DLT 9x9, 12x12 symmetric matrixes, and run UDU', UDV'
	
    D. gen DLT 12x12 for n points, and run UDU', UDV'
    
warning: only tested with visual studio 10, but would be fine in linux (clock.hpp changes on both systems) this test requires opencv and uses opencv "lapack.cpp" functions; almost all functions of the interest here, have been renamed with a posfix underscore and located in blas.hpp (math) and udut_givens.hpp, udvt_givens.hpp (for decompositions) in order to be called, tested, and compiled inline and with optimization flags altogether. other opencv structures and functions not used, eg. cv::Mat, cv::Size, cv::InputArray are using the normal cv:: interface.

USE: #define USE_FP32 if you want to used floats instead of doubles

i used in a project with 2 configs were _DEBUG i set no optimization flags and RELEASE full optimization; so if that is the case, it will output the strings "VC10 not opt" and "VC10 OX" in the plots;

the output are 4 plots that are appended to "res.txt" that you can insert them all in "plot_test.m" to see the results (uses uiaxes.m).

the result of those tests - at least in my machine, running windows10/VC10 - (B) UDU' has a clear advantage over UDV' on large matrices; (D) but in small matrices as in the DLT 12x12 case, the results are mixed: UDV' is better in 64bits both full optimization and 32-64bit without, and UDU' is better with optimization on 32bits. (A) for the generation of the symmetric X'X DLT system is always better to do it manually in a loop and copy the upper triangle instead of creating the DLT X system in a loop and calling for doing a product with the transpose of X.
