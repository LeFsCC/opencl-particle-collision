#pragma once
#include <cl/cl.h>
#include <stdlib.h>
#define oclCheckErrorEX(a, b, c) __oclCheckErrorEX(a, b, c, __FILE__ , __LINE__) 
#define oclCheckError(a, b) oclCheckErrorEX(a, b, 0)
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)

inline void __oclCheckErrorEX(cl_int iSample, cl_int iReference, void (*pCleanup)(int), const char* cFile, const int iLine) {
    if (iReference != iSample) {
        iSample = (iSample == 0) ? -9999 : iSample;

        if (pCleanup != NULL) {
            pCleanup(iSample);
        }
        else {
            exit(iSample);
        }
    }
}
