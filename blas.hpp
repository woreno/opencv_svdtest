/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef VL_BLAS
#define VL_BLAS

// C
	#include <string.h>
	#include <limits.h>
	#include <stdarg.h>
	#include <vector>
	#include <math.h>
	#include <float.h>
	#include <string>
	#include <time.h>
// intrisecas intel SSEX/AVX
	#include <xmmintrin.h> 
	#include <emmintrin.h>
	#include <immintrin.h>
#include <opencv2/opencv.hpp>

// math consts
#define VL_M_E        2.71828182845904523536	// e	
#define VL_M_E_1	  1.71828182846				// e-1
#define VL_M_LOG2E    1.44269504088896340736	// log(e)/log(2)
#define VL_M_PI       3.14159265358979323846	// pi
#define VL_M_PI3O2	  4.712388980384690			// (3/2)*pi
#define VL_M_PI2O3	  2.094395102393195			// (2/3)*pi
#define VL_M_PI4O3	  4.188790204786391			// (4/3)*pi
#define VL_M_PI2      6.283185307179586			// (2)*pi
#define VL_M_PIO2     1.57079632679489661923	// (1/2)*pi
#define VL_M_PHI      1.61803398875
#define VL_M_180OPI	  57.295779513082323		// 180/pi
#define VL_M_PIO180   0.017453292519943			// pi/1280
#define VL_M_SQRT2    1.41421356237309504880	// sqrt(2)
#define VL_M_SQRT2O2  0.707106781186547524401	// sqrt(2)/2
#define VL_M_INVSQRT2 0.70710678118				// (1/sqrt(2))
#define	VL_M_ROUND(A) ((((A)-floor(A))<0.5f)? floor(A):ceil(A)) 
#define VL_SRAND	  srand(time(0));
#define VL_RAND(X)	  ((rand()%X))
#define VL_RAD2DEG(X) ((X)*VL_M_180OPI)
#define VL_DEG2RAD(X) ((X)*VL_M_PIO180)



// begin OPENCV code... see license.txt
// slighty modified to not use the OPENCV-API
// excepting the basic structures

template<typename _Tp> static inline _Tp hypot_(_Tp a, _Tp b)
{
    a = std::abs(a);
    b = std::abs(b);
    if( a > b )
    {
        b /= a;
        return a*std::sqrt(1 + b*b);
    }
    if( b > 0 )
    {
        a /= b;
        return b*std::sqrt(1 + a*a);
    }
    return 0;
}


template<typename _Tp> static inline _Tp* alignPtr_(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

template<typename T> struct VBLAS
{
    int dot(const T*, const T*, int, T*) const { return 0; }
    int givens(T*, T*, int, T, T) const { return 0; }
    int givensx(T*, T*, int, T, T, T*, T*) const { return 0; }
};


template<> inline int VBLAS<float>::dot(const float* a, const float* b, int n, float* result) const
{
    if( n < 8 )
        return 0;
    int k = 0;
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    for( ; k <= n - 8; k += 8 )
    {
        __m128 a0 = _mm_load_ps(a + k), a1 = _mm_load_ps(a + k + 4);
        __m128 b0 = _mm_load_ps(b + k), b1 = _mm_load_ps(b + k + 4);

        s0 = _mm_add_ps(s0, _mm_mul_ps(a0, b0));
        s1 = _mm_add_ps(s1, _mm_mul_ps(a1, b1));
    }
    s0 = _mm_add_ps(s0, s1);
    float sbuf[4];
    _mm_storeu_ps(sbuf, s0);
    *result = sbuf[0] + sbuf[1] + sbuf[2] + sbuf[3];
    return k;
}


template<> inline int VBLAS<float>::givens(float* a, float* b, int n, float c, float s) const
{
    if( n < 4 )
        return 0;
    int k = 0;
    __m128 c4 = _mm_set1_ps(c), s4 = _mm_set1_ps(s);
    for( ; k <= n - 4; k += 4 )
    {
        __m128 a0 = _mm_load_ps(a + k);
        __m128 b0 = _mm_load_ps(b + k);
        __m128 t0 = _mm_add_ps(_mm_mul_ps(a0, c4), _mm_mul_ps(b0, s4));
        __m128 t1 = _mm_sub_ps(_mm_mul_ps(b0, c4), _mm_mul_ps(a0, s4));
        _mm_store_ps(a + k, t0);
        _mm_store_ps(b + k, t1);
    }
    return k;
}


template<> inline int VBLAS<float>::givensx(float* a, float* b, int n, float c, float s,
                                             float* anorm, float* bnorm) const
{
    if( n < 4 )
        return 0;
    int k = 0;
    __m128 c4 = _mm_set1_ps(c), s4 = _mm_set1_ps(s);
    __m128 sa = _mm_setzero_ps(), sb = _mm_setzero_ps();
    for( ; k <= n - 4; k += 4 )
    {
        __m128 a0 = _mm_load_ps(a + k);
        __m128 b0 = _mm_load_ps(b + k);
        __m128 t0 = _mm_add_ps(_mm_mul_ps(a0, c4), _mm_mul_ps(b0, s4));
        __m128 t1 = _mm_sub_ps(_mm_mul_ps(b0, c4), _mm_mul_ps(a0, s4));
        _mm_store_ps(a + k, t0);
        _mm_store_ps(b + k, t1);
        sa = _mm_add_ps(sa, _mm_mul_ps(t0, t0));
        sb = _mm_add_ps(sb, _mm_mul_ps(t1, t1));
    }
    float abuf[4], bbuf[4];
    _mm_storeu_ps(abuf, sa);
    _mm_storeu_ps(bbuf, sb);
    *anorm = abuf[0] + abuf[1] + abuf[2] + abuf[3];
    *bnorm = bbuf[0] + bbuf[1] + bbuf[2] + bbuf[3];
    return k;
}


template<> inline int VBLAS<double>::dot(const double* a, const double* b, int n, double* result) const
{
    if( n < 4 )
        return 0;
    int k = 0;
    __m128d s0 = _mm_setzero_pd(), s1 = _mm_setzero_pd();
    for( ; k <= n - 4; k += 4 )
    {
        __m128d a0 = _mm_load_pd(a + k), a1 = _mm_load_pd(a + k + 2);
        __m128d b0 = _mm_load_pd(b + k), b1 = _mm_load_pd(b + k + 2);

        s0 = _mm_add_pd(s0, _mm_mul_pd(a0, b0));
        s1 = _mm_add_pd(s1, _mm_mul_pd(a1, b1));
    }
    s0 = _mm_add_pd(s0, s1);
    double sbuf[2];
    _mm_storeu_pd(sbuf, s0);
    *result = sbuf[0] + sbuf[1];
    return k;
}


template<> inline int VBLAS<double>::givens(double* a, double* b, int n, double c, double s) const
{
    int k = 0;
    __m128d c2 = _mm_set1_pd(c), s2 = _mm_set1_pd(s);
    for( ; k <= n - 2; k += 2 )
    {
        __m128d a0 = _mm_load_pd(a + k);
        __m128d b0 = _mm_load_pd(b + k);
        __m128d t0 = _mm_add_pd(_mm_mul_pd(a0, c2), _mm_mul_pd(b0, s2));
        __m128d t1 = _mm_sub_pd(_mm_mul_pd(b0, c2), _mm_mul_pd(a0, s2));
        _mm_store_pd(a + k, t0);
        _mm_store_pd(b + k, t1);
    }
    return k;
}


template<> inline int VBLAS<double>::givensx(double* a, double* b, int n, double c, double s,
                                              double* anorm, double* bnorm) const
{
    int k = 0;
    __m128d c2 = _mm_set1_pd(c), s2 = _mm_set1_pd(s);
    __m128d sa = _mm_setzero_pd(), sb = _mm_setzero_pd();
    for( ; k <= n - 2; k += 2 )
    {
        __m128d a0 = _mm_load_pd(a + k);
        __m128d b0 = _mm_load_pd(b + k);
        __m128d t0 = _mm_add_pd(_mm_mul_pd(a0, c2), _mm_mul_pd(b0, s2));
        __m128d t1 = _mm_sub_pd(_mm_mul_pd(b0, c2), _mm_mul_pd(a0, s2));
        _mm_store_pd(a + k, t0);
        _mm_store_pd(b + k, t1);
        sa = _mm_add_pd(sa, _mm_mul_pd(t0, t0));
        sb = _mm_add_pd(sb, _mm_mul_pd(t1, t1));
    }
    double abuf[2], bbuf[2];
    _mm_storeu_pd(abuf, sa);
    _mm_storeu_pd(bbuf, sb);
    *anorm = abuf[0] + abuf[1];
    *bnorm = bbuf[0] + bbuf[1];
    return k;
}



inline void cvCompleteSymm_( CvMat* matrix, int LtoR )
{
    cv::Mat m(matrix);
    cv::completeSymm( m, LtoR != 0 );
}

inline void completeSymm_( cv::InputOutputArray _m, bool LtoR )
{
	using namespace cv;
    Mat m = _m.getMat();
    size_t step = m.step, esz = m.elemSize();
    CV_Assert( m.dims <= 2 && m.rows == m.cols );

    int rows = m.rows;
    int j0 = 0, j1 = rows;

    uchar* data = m.data;
    for( int i = 0; i < rows; i++ )
    {
        if( !LtoR ) j1 = i; else j0 = i+1;
        for( int j = j0; j < j1; j++ )
            memcpy(data + (i*step + j*esz), data + (j*step + i*esz), esz);
    }
}

typedef void (*MulTransposedFunc)(const cv::Mat& src, cv::Mat& dst, const cv::Mat& delta, double scale);

template<typename sT, typename dT> static void
MulTransposedR( const cv::Mat& srcmat, cv::Mat& dstmat, const cv::Mat& deltamat, double scale )
{
	using namespace cv;
    int i, j, k;
    const sT* src = (const sT*)srcmat.data;
    dT* dst = (dT*)dstmat.data;
    const dT* delta = (const dT*)deltamat.data;
    size_t srcstep = srcmat.step/sizeof(src[0]);
    size_t dststep = dstmat.step/sizeof(dst[0]);
    size_t deltastep = deltamat.rows > 1 ? deltamat.step/sizeof(delta[0]) : 0;
    int delta_cols = deltamat.cols;
    Size size = srcmat.size();
    dT* tdst = dst;
    dT* col_buf = 0;
    dT* delta_buf = 0;
    int buf_size = size.height*sizeof(dT);
    AutoBuffer<uchar> buf;

    if( delta && delta_cols < size.width )
    {
        assert( delta_cols == 1 );
        buf_size *= 5;
    }
    buf.allocate(buf_size);
    col_buf = (dT*)(uchar*)buf;

    if( delta && delta_cols < size.width )
    {
        delta_buf = col_buf + size.height;
        for( i = 0; i < size.height; i++ )
            delta_buf[i*4] = delta_buf[i*4+1] =
                delta_buf[i*4+2] = delta_buf[i*4+3] = delta[i*deltastep];
        delta = delta_buf;
        deltastep = deltastep ? 4 : 0;
    }

    if( !delta )
        for( i = 0; i < size.width; i++, tdst += dststep )
        {
            for( k = 0; k < size.height; k++ )
                col_buf[k] = src[k*srcstep+i];

            for( j = i; j <= size.width - 4; j += 4 )
            {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                const sT *tsrc = src + j;

                for( k = 0; k < size.height; k++, tsrc += srcstep )
                {
                    double a = col_buf[k];
                    s0 += a * tsrc[0];
                    s1 += a * tsrc[1];
                    s2 += a * tsrc[2];
                    s3 += a * tsrc[3];
                }

                tdst[j] = (dT)(s0*scale);
                tdst[j+1] = (dT)(s1*scale);
                tdst[j+2] = (dT)(s2*scale);
                tdst[j+3] = (dT)(s3*scale);
            }

            for( ; j < size.width; j++ )
            {
                double s0 = 0;
                const sT *tsrc = src + j;

                for( k = 0; k < size.height; k++, tsrc += srcstep )
                    s0 += (double)col_buf[k] * tsrc[0];

                tdst[j] = (dT)(s0*scale);
            }
        }
    else
        for( i = 0; i < size.width; i++, tdst += dststep )
        {
            if( !delta_buf )
                for( k = 0; k < size.height; k++ )
                    col_buf[k] = src[k*srcstep+i] - delta[k*deltastep+i];
            else
                for( k = 0; k < size.height; k++ )
                    col_buf[k] = src[k*srcstep+i] - delta_buf[k*deltastep];

            for( j = i; j <= size.width - 4; j += 4 )
            {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                const sT *tsrc = src + j;
                const dT *d = delta_buf ? delta_buf : delta + j;

                for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                {
                    double a = col_buf[k];
                    s0 += a * (tsrc[0] - d[0]);
                    s1 += a * (tsrc[1] - d[1]);
                    s2 += a * (tsrc[2] - d[2]);
                    s3 += a * (tsrc[3] - d[3]);
                }

                tdst[j] = (dT)(s0*scale);
                tdst[j+1] = (dT)(s1*scale);
                tdst[j+2] = (dT)(s2*scale);
                tdst[j+3] = (dT)(s3*scale);
            }

            for( ; j < size.width; j++ )
            {
                double s0 = 0;
                const sT *tsrc = src + j;
                const dT *d = delta_buf ? delta_buf : delta + j;

                for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                    s0 += (double)col_buf[k] * (tsrc[0] - d[0]);

                tdst[j] = (dT)(s0*scale);
            }
        }
}

template<typename sT, typename dT> static void
MulTransposedL( const cv::Mat& srcmat, cv::Mat& dstmat, const cv::Mat& deltamat, double scale )
{
	using namespace cv;
    int i, j, k;
    const sT* src = (const sT*)srcmat.data;
    dT* dst = (dT*)dstmat.data;
    const dT* delta = (const dT*)deltamat.data;
    size_t srcstep = srcmat.step/sizeof(src[0]);
    size_t dststep = dstmat.step/sizeof(dst[0]);
    size_t deltastep = deltamat.rows > 1 ? deltamat.step/sizeof(delta[0]) : 0;
    int delta_cols = deltamat.cols;
    Size size = srcmat.size();
    dT* tdst = dst;

    if( !delta )
        for( i = 0; i < size.height; i++, tdst += dststep )
            for( j = i; j < size.height; j++ )
            {
                double s = 0;
                const sT *tsrc1 = src + i*srcstep;
                const sT *tsrc2 = src + j*srcstep;

                for( k = 0; k <= size.width - 4; k += 4 )
                    s += (double)tsrc1[k]*tsrc2[k] + (double)tsrc1[k+1]*tsrc2[k+1] +
                         (double)tsrc1[k+2]*tsrc2[k+2] + (double)tsrc1[k+3]*tsrc2[k+3];
                for( ; k < size.width; k++ )
                    s += (double)tsrc1[k] * tsrc2[k];
                tdst[j] = (dT)(s*scale);
            }
    else
    {
        dT delta_buf[4];
        int delta_shift = delta_cols == size.width ? 4 : 0;
        cv::AutoBuffer<uchar> buf(size.width*sizeof(dT));
        dT* row_buf = (dT*)(uchar*)buf;

        for( i = 0; i < size.height; i++, tdst += dststep )
        {
            const sT *tsrc1 = src + i*srcstep;
            const dT *tdelta1 = delta + i*deltastep;

            if( delta_cols < size.width )
                for( k = 0; k < size.width; k++ )
                    row_buf[k] = tsrc1[k] - tdelta1[0];
            else
                for( k = 0; k < size.width; k++ )
                    row_buf[k] = tsrc1[k] - tdelta1[k];

            for( j = i; j < size.height; j++ )
            {
                double s = 0;
                const sT *tsrc2 = src + j*srcstep;
                const dT *tdelta2 = delta + j*deltastep;
                if( delta_cols < size.width )
                {
                    delta_buf[0] = delta_buf[1] =
                        delta_buf[2] = delta_buf[3] = tdelta2[0];
                    tdelta2 = delta_buf;
                }
                for( k = 0; k <= size.width-4; k += 4, tdelta2 += delta_shift )
                    s += (double)row_buf[k]*(tsrc2[k] - tdelta2[0]) +
                         (double)row_buf[k+1]*(tsrc2[k+1] - tdelta2[1]) +
                         (double)row_buf[k+2]*(tsrc2[k+2] - tdelta2[2]) +
                         (double)row_buf[k+3]*(tsrc2[k+3] - tdelta2[3]);
                for( ; k < size.width; k++, tdelta2++ )
                    s += (double)row_buf[k]*(tsrc2[k] - tdelta2[0]);
                tdst[j] = (dT)(s*scale);
            }
        }
    }
}

void mulTransposed_( cv::InputArray _src, cv::OutputArray _dst, bool ata,
                     cv::InputArray _delta, double scale, int dtype )
{
	using namespace cv;
    Mat src = _src.getMat(), delta = _delta.getMat();
    const int gemm_level = 100; // boundary above which GEMM is faster.
    int stype = src.type();
    dtype = std::max(std::max(CV_MAT_DEPTH(dtype >= 0 ? dtype : stype), delta.depth()), CV_32F);
    CV_Assert( src.channels() == 1 );

    if( delta.data )
    {
        CV_Assert( delta.channels() == 1 &&
            (delta.rows == src.rows || delta.rows == 1) &&
            (delta.cols == src.cols || delta.cols == 1));
        if( delta.type() != dtype )
            delta.convertTo(delta, dtype);
    }

    int dsize = ata ? src.cols : src.rows;
    _dst.create( dsize, dsize, dtype );
    Mat dst = _dst.getMat();

    if( src.data == dst.data || (stype == dtype &&
        (dst.cols >= gemm_level && dst.rows >= gemm_level &&
         src.cols >= gemm_level && src.rows >= gemm_level)))
    {
        Mat src2;
        const Mat* tsrc = &src;
        if( delta.data )
        {
            if( delta.size() == src.size() )
                subtract( src, delta, src2 );
            else
            {
                repeat(delta, src.rows/delta.rows, src.cols/delta.cols, src2);
                subtract( src, src2, src2 );
            }
            tsrc = &src2;
        }
        gemm( *tsrc, *tsrc, scale, Mat(), 0, dst, ata ? GEMM_1_T : GEMM_2_T );
    }
    else
    {
        MulTransposedFunc func = 0;
        if(stype == CV_8U && dtype == CV_32F)
        {
            if(ata)
                func = MulTransposedR<uchar,float>;
            else
                func = MulTransposedL<uchar,float>;
        }
        else if(stype == CV_8U && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<uchar,double>;
            else
                func = MulTransposedL<uchar,double>;
        }
        else if(stype == CV_16U && dtype == CV_32F)
        {
            if(ata)
                func = MulTransposedR<ushort,float>;
            else
                func = MulTransposedL<ushort,float>;
        }
        else if(stype == CV_16U && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<ushort,double>;
            else
                func = MulTransposedL<ushort,double>;
        }
        else if(stype == CV_16S && dtype == CV_32F)
        {
            if(ata)
                func = MulTransposedR<short,float>;
            else
                func = MulTransposedL<short,float>;
        }
        else if(stype == CV_16S && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<short,double>;
            else
                func = MulTransposedL<short,double>;
        }
        else if(stype == CV_32F && dtype == CV_32F)
        {
            if(ata)
                func = MulTransposedR<float,float>;
            else
                func = MulTransposedL<float,float>;
        }
        else if(stype == CV_32F && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<float,double>;
            else
                func = MulTransposedL<float,double>;
        }
        else if(stype == CV_64F && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<double,double>;
            else
                func = MulTransposedL<double,double>;
        }
        if( !func )
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src, dst, delta, scale );
        completeSymm( dst, false );
    }
}


void
cvMulTransposed_( const CvArr* srcarr, CvArr* dstarr,
                 int order, const CvArr* deltaarr CV_DEFAULT(NULL), double scale CV_DEFAULT(1.) )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0, delta;
    if( deltaarr )
        delta = cv::cvarrToMat(deltaarr);
    mulTransposed_( src, dst, order != 0, delta, scale, dst.type());
    if( dst.data != dst0.data )
        dst.convertTo(dst0, dst0.type());
}


#endif 