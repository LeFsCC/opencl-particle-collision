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
static const unsigned int LOCAL_SIZE_LIMIT = 512U;

static cl_kernel ckIntegrate, ckCalcHash, ckMemset, ckFindCellBoundsAndReorder, ckCollide;


extern "C" void startupOpenCL();
extern "C" void allocateArray(memHandle_t * memObj, size_t size);
extern "C" void freeArray(memHandle_t memObj);
extern "C" void copyArrayFromDevice(void* hostPtr, const memHandle_t memObj, size_t size);
extern "C" void copyArrayToDevice(memHandle_t memObj, const void* hostPtr, size_t offset, size_t size);
extern "C" void setParameters(simParams_t * m_params);
extern "C" void setParametersHost(simParams_t * host_params);
extern "C" void closeParticles(void);
extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue);
extern "C" void integrateSystem(memHandle_t d_Pos, memHandle_t d_Vel, float deltaTime, uint numParticles);
extern "C" void calcHash(memHandle_t d_Hash, memHandle_t d_Index, memHandle_t d_Pos, int numParticles);
static void memsetOCL(memHandle_t d_Data, uint val, uint N);
extern "C" void findCellBoundsAndReorder(memHandle_t d_CellStart, memHandle_t d_CellEnd, memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel, memHandle_t d_Hash, memHandle_t d_Index, memHandle_t d_Pos, memHandle_t d_Vel,
    uint numParticles, uint numCells);
extern "C" void collide(memHandle_t d_Vel, memHandle_t d_ReorderedPos, memHandle_t d_ReorderedVel, memHandle_t d_Index,
    memHandle_t d_CellStart, memHandle_t d_CellEnd, uint   numParticles, uint   numCells);
static size_t uSnap(size_t a, size_t b);


extern"C" void bitonicSort(cl_command_queue cqCommandQueue, cl_mem d_DstKey, cl_mem d_DstVal, cl_mem d_SrcKey,
    cl_mem d_SrcVal, unsigned int batch, unsigned int arrayLength, unsigned int dir);


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
