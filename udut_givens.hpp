#ifndef VL_UDUt
#define VL_UDUt
#include "blas.hpp"

// begin OPENCV code... see license.txt 
// slighty modified to not use the OPENCV-API
// excepting the basic structures

#define CV_DEBUG_JACOBI(S)
#define CV_DEBUG_JACOBI_PEEK(S)


template<typename _Tp> bool
JacobiImpl_( _Tp* A, size_t astep, 
			 _Tp* W, 
			 _Tp* V, size_t vstep, int n, uchar* buf )
{
    const _Tp eps = std::numeric_limits<_Tp>::epsilon();
    int i, j, k, m;

    astep /= sizeof(A[0]);
    if( V )
    {
        vstep /= sizeof(V[0]);
        for( i = 0; i < n; i++ )
        {
            for( j = 0; j < n; j++ )
                V[i*vstep + j] = (_Tp)0;
            V[i*vstep + i] = (_Tp)1;
        }
    }

    int iters, maxIters = n*n*30;

    int* indR = (int*)alignPtr_(buf, sizeof(int));
    int* indC = indR + n;
    _Tp mv = (_Tp)0;

    for( k = 0; k < n; k++ )
    {
		// copiar valor da diagonal para o W
        W[k] = A[(astep + 1)*k];
        if( k < n - 1 )
        {
			// em cada linha, à direita da diagonal...
            for( m = k+1, mv = std::abs(A[astep*k + m]), 
				 i = k+2; i < n; i++ )
            {
                _Tp val = std::abs(A[astep*k+i]);
                if( mv < val )
                    mv = val, m = i;
            }
			// encontrar o maior valor absoluto mv, e o
			// respetivo indice m
            indR[k] = m;
			// guardar este indice por linha em indR
        }
        if( k > 0 )
        {
			// encima da diagonal, procurar o maior 
			// valor para cada coluna ...
            for( m = 0, 
				 mv = std::abs(A[k]), 
				 i = 1; 
			     i < k; 
				 i++ )
            {
                _Tp val = std::abs(A[astep*i+k]);
                if( mv < val )
                    mv = val, m = i;
            }
			// ... e guardar o indice em indC
            indC[k] = m;
        }
    }
	
	CV_DEBUG_JACOBI_PEEK("inicio")

    if( n > 1 ) for( iters = 0; iters < maxIters; iters++ )
    {
        // pesquisar nos indices de indR o maior;
		// tomando como referência este valor
		// voltar a pesquisar em indC se existe um maior
		// redefinir para o maior se existir,
		// o pivot é dado por k (linha), l (coluna)
        for( 
			 k = 0, 
			 mv = std::abs(A[indR[0]]), 
			 i = 1; i < n-1; i++ 
		   )
        {
            _Tp val = std::abs(A[astep*i + indR[i]]);
            if( mv < val )
                mv = val, k = i;
        }

        int l = indR[k];
        for( i = 1; i < n; i++ )
        {
            _Tp val = std::abs(A[astep*indC[i] + i]);
            if( mv < val )
                mv = val, k = indC[i], l = i;
        }

		// p é valo atual do pivot
        _Tp p = A[astep*k + l];
		//
        if( std::abs(p) <= eps )
            break;

		// calcular o seno e coseno, que apareceria
		// na matriz de Givens para anular o valor do pivot
        _Tp y = (_Tp)((W[l] - W[k])*0.5);
        _Tp t = std::abs(y) + hypot(p, y); // novo dx do triangulo
        _Tp w = hypot(p, t);			   // nova hipotenusa "norma de t,p"
        _Tp c = t/w;					   // normalização, componente x: cos
        _Tp s = p/w;					   // normalização, componente y: sin
		
		_Tp u = ( p/t ) * p;
        if ( y < 0 )
            s = -s, u = -u;

		// zerar pivot
        A[  astep*k + l ] = 0;

		//atualizar diagonais
        W[k] -= u; // pivot kk (ii)
        W[l] += u; // pivot ll (jj)

        _Tp a0, b0;

#undef rotate
#define rotate(v0, v1)    \
		a0 = v0,          \
		b0 = v1,          \
		v0 = a0*c - b0*s, \
		v1 = a0*s + b0*c  

	// rodar linhas e colunas de A
        //     linha i:[0..k-1] coluna k 
		// <-> linha i:[0..k-1] coluna l
        for( i = 0; i < k; i++ )
            rotate(A[astep*i+k], A[astep*i+l]);

		//     linha k, coluna i:[k+1,l-1] 
		// <-> linha i[k+1,l-1], coluna l
        for( i = k+1; i < l; i++ )
            rotate(A[astep*k+i], A[astep*i+l]);

		//     linha k, coluna i:[l+1..n-1]
		// <-> linha l, coluna i:[l+1..n-1]
        for( i = l+1; i < n; i++ )
            rotate(A[astep*k+i], A[astep*l+i]);

	//rodar vetores proprios
		//     linha k, coluna i:[0..n-1] <->
		//     linha l, coluna i:[0..n-1]
        if( V )
            for( i = 0; i < n; i++ )
                rotate(V[vstep*k+i], V[vstep*l+i]);
		
		CV_DEBUG_JACOBI("apos rotacao")

#undef rotate

		// atualizar o vetor de máximos para as linhas/colunas
        for( j = 0; j < 2; j++ )
        {
            int idx = j == 0 ? k : l;
            if( idx < n - 1 )
            {
                for( m = idx+1, 
					 mv = std::abs(A[astep*idx + m]), 
					 i = idx+2; i < n; i++ )
                {
                    _Tp val = std::abs(A[astep*idx+i]);
                    if( mv < val )
                        mv = val, m = i;
                }
                indR[idx] = m;
            }
            if( idx > 0 )
            {
                for( m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++ )
                {
                    _Tp val = std::abs(A[astep*i+idx]);
                    if( mv < val )
                        mv = val, m = i;
                }
                indC[idx] = m;
            }
        }
		
		CV_DEBUG_JACOBI("apos update de maximos")
    }

    // sort eigenvalues & eigenvectors
    for( k = 0; k < n-1; k++ )
    {
        m = k;
        for( i = k+1; i < n; i++ )
        {
            if( W[m] < W[i] )
                m = i;
        }
        if( k != m )
        {
            std::swap(W[m], W[k]);
            if( V )
                for( i = 0; i < n; i++ )
                    std::swap(V[vstep*m + i], V[vstep*k + i]);
        }
    }
	
	CV_DEBUG_JACOBI_PEEK("fim, apos sort")
    return true;
}


inline bool Jacobi( float* S, size_t sstep, float* e, float* E, size_t estep, int n, uchar* buf )
{
    return JacobiImpl_(S, sstep, e, E, estep, n, buf);
}

inline bool Jacobi( double* S, size_t sstep, double* e, double* E, size_t estep, int n, uchar* buf )
{
    return JacobiImpl_(S, sstep, e, E, estep, n, buf);
}

bool udut( cv::InputArray _src, bool computeEvects, cv::OutputArray _evals, 
		   cv::OutputArray _evects )
{
	using namespace cv;
    Mat src = _src.getMat();
    int type = src.type();
    int n = src.rows;

    CV_Assert( src.rows == src.cols );
    CV_Assert (type == CV_32F || type == CV_64F);

    Mat v;
    if( computeEvects )
    {
        _evects.create(n, n, type);
        v = _evects.getMat();
    }

    size_t elemSize = src.elemSize(), astep = alignSize(n*elemSize, 16);
    AutoBuffer<uchar> buf(n*astep + n*5*elemSize + 32);
    uchar* ptr = alignPtr_<uchar>((uchar*)buf, 16);
    Mat a(n, n, type, ptr, astep), w(n, 1, type, ptr + astep*n);
    ptr += astep*n + elemSize*n;
    src.copyTo(a);
    bool ok = type == CV_32F ?
        Jacobi(a.ptr<float>(), a.step, w.ptr<float>(), v.ptr<float>(), v.step, n, ptr) :
        Jacobi(a.ptr<double>(), a.step, w.ptr<double>(), v.ptr<double>(), v.step, n, ptr);

    w.copyTo(_evals);
    return ok;
}


















#endif