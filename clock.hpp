#ifndef VL_CLOCK
#define VL_CLOCK
#include <vector>
#include <string>
#include <math.h>

#if (defined(_WIN64) || defined(_WIN32))
	#define VL_SYSTEM_WINDOWS
#else
		#if defined(_AIX) \
			|| defined(__APPLE__)
			|| defined(__FreeBSD__) || defined(__DragonFly__) \
			|| defined(__FreeBSD_kernel__) \
			|| defined(__GNU__)
			|| defined(__linux__) \
			|| defined(__NetBSD__) \
			|| defined(__OpenBSD__) \
			|| defined(__QNXNTO__) \
			|| defined(sun) || defined(__sun)
			|| defined(unix) || defined(__unix) || defined(__unix__)
			#define VL_SYSTEM_UNIX
		#endif
#endif

#define VL_DECL_CLOCKPRI()		double VL_CLOCK_MEAN = 0, VL_CLOCK_DP = 0; std::vector<double> VL_CLOCK_TIMES; std::string VL_CLOCK_NAME; int VL_TEST_N1, VL_TEST_N2; 

#ifdef VL_SYSTEM_UNIX
    #include <time.h>
	#define	VL_DECL_CLOCK()			VL_DECL_CLOCKPRI()  clock_t   VL_CLOCK_T0, VL_CLOCK_T1;
	#define VL_TIC()				VL_CLOCK_T0 = clock();
	#define VL_TOC()				VL_CLOCK_T1 = clock();
	#define VL_GET_CLOCK_S()		(((double)(VL_CLOCK_T1-VL_CLOCK_T0))/(double)CLOCKS_PER_SEC)
	#define VL_GET_CLOCK_MS()		(VL_GET_CLOCK_S()*1e-3)
	#define VL_GET_CLOCK_US()		(VL_GET_CLOCK_S()*1e-6) 
#else
	#define WIN32_LEAN_AND_MEAN
	#include <Windows.h>
	#define	VL_DECL_CLOCK()			VL_DECL_CLOCKPRI() LARGE_INTEGER VL_CLOCK_F; LARGE_INTEGER VL_CLOCK_T0, VL_CLOCK_T1;  
	#define VL_TIC()				{ if (::QueryPerformanceFrequency(&VL_CLOCK_F) == FALSE) throw "ouch!"; if (::QueryPerformanceCounter(&VL_CLOCK_T0) == FALSE) throw "foo"; }
	#define VL_TOC()				{ if (::QueryPerformanceCounter(&VL_CLOCK_T1) == FALSE) throw "ouch!";}
	#define VL_GET_CLOCK_S()		((double)(static_cast<double>(VL_CLOCK_T1.QuadPart - VL_CLOCK_T0.QuadPart) / VL_CLOCK_F.QuadPart))
	#define VL_GET_CLOCK_MS()		(1e3*((double)(static_cast<double>(VL_CLOCK_T1.QuadPart - VL_CLOCK_T0.QuadPart) / VL_CLOCK_F.QuadPart)))
	#define VL_GET_CLOCK_US()		(1e6*((double)(static_cast<double>(VL_CLOCK_T1.QuadPart - VL_CLOCK_T0.QuadPart) / VL_CLOCK_F.QuadPart)))

	#define	VL_DECL_CLOCK_(ID)			LARGE_INTEGER VL_CLOCK_F##ID; LARGE_INTEGER VL_CLOCK_T0##ID, VL_CLOCK_T1##ID;  
	#define VL_TIC_(ID)					{ if (::QueryPerformanceFrequency(&VL_CLOCK_F##ID) == FALSE) throw "foo"; if (::QueryPerformanceCounter(&VL_CLOCK_T0##ID) == FALSE) throw "foo"; }
	#define VL_TOC_(ID)					{ if (::QueryPerformanceCounter(&VL_CLOCK_T1##ID) == FALSE) throw "foo";}
	#define VL_GET_CLOCK_MS_(ID)		(1e3*((double)(static_cast<double>(VL_CLOCK_T1##ID.QuadPart - VL_CLOCK_T0##ID.QuadPart) / VL_CLOCK_F##ID.QuadPart)))
#endif 

	#define	VL_CLOCK_REP_TIC_STATS(NTESTS1,NTESTS2) VL_TEST_N1=NTESTS1; VL_TEST_N2=NTESTS2; VL_CLOCK_TIMES.resize(VL_TEST_N1); 

	#define	VL_CLOCK_REP_SILTIC(NTESTS1,NTESTS2)  VL_CLOCK_REP_TIC_STATS(NTESTS1,NTESTS2) for (int VL_CLOCKI=0; VL_CLOCKI< VL_TEST_N1; VL_CLOCKI++) { VL_TIC()  for (int VL_CLOCKJ=0; VL_CLOCKJ< VL_TEST_N2; VL_CLOCKJ++)  {

	#define VL_CLOCK_REP_TOC_STATS() } VL_TOC()								    \
		VL_CLOCK_TIMES[VL_CLOCKI]=VL_GET_CLOCK_US()/double(VL_TEST_N2); }		\
		VL_CLOCK_MEAN = 0, VL_CLOCK_DP = 0;  									\
		for (int VL_CLOCKI=0; VL_CLOCKI< VL_CLOCK_TIMES.size(); VL_CLOCKI++)  	\
			VL_CLOCK_MEAN+= VL_CLOCK_TIMES[VL_CLOCKI];  						\
		VL_CLOCK_MEAN/=(double)VL_CLOCK_TIMES.size();  							\
		for (int VL_CLOCKI=0; VL_CLOCKI< VL_CLOCK_TIMES.size(); VL_CLOCKI++)  	\
			VL_CLOCK_DP += pow((VL_CLOCK_TIMES[VL_CLOCKI]-VL_CLOCK_MEAN),2);  	\
		VL_CLOCK_DP = sqrt(VL_CLOCK_DP)/(double)VL_CLOCK_TIMES.size();			


	// silencioso
	#define VL_CLOCK_REP_SILTOC() VL_CLOCK_REP_TOC_STATS()  


#endif