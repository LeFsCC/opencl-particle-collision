#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <cl/opencl.h>
#pragma warning( disable : 4996 )
#include <cl/cl.h>
#include "SphereSystem.h"
#include <fstream>
#define oclCheckErrorEX(a, b, c) __oclCheckErrorEX(a, b, c, __FILE__ , __LINE__) 
#define oclCheckError(a, b) oclCheckErrorEX(a, b, 0) 
typedef cl_mem memHandle_t;
typedef unsigned int uint;
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)

extern "C" void startupOpenCL();
extern "C" void shutdownOpenCL(void);
extern "C" void allocateArray(memHandle_t * memObj, size_t size);
extern "C" void freeArray(memHandle_t memObj);
extern "C" void copyArrayFromDevice(void* hostPtr, const memHandle_t memObj, unsigned int vbo, size_t size);
extern "C" void copyArrayToDevice(memHandle_t memObj, const void* hostPtr, size_t offset, size_t size);
extern "C" void setParameters(simParams_t * m_params);
extern "C" void setParametersHost(simParams_t * host_params);
extern "C" void closeParticles(void);

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

#endif UTILS_H
