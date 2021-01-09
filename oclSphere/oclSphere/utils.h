#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <cl/opencl.h>
#pragma warning( disable : 4996 )
#include <cl/cl.h>
#include "SphereSystem.h"
#include <fstream>
#define oclCheckError(a, b) __oclCheckErrorEX(a, b, 0, __FILE__ , __LINE__)
typedef cl_mem cl_mem;
typedef unsigned int uint;
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)
static const unsigned int LOCAL_SIZE_LIMIT = 512U;

static cl_kernel ckIntegrate, ckCalcHash, ckMemset, ckFindCellBoundsAndReorder, ckCollide;


extern "C" void prepareOpenCLPlatform();
extern "C" void allocateArray(cl_mem * memObj, size_t size);
extern "C" void freeArray(cl_mem memObj);
extern "C" void copyArrayFromDevice(void* hostPtr, const cl_mem memObj, size_t size);
extern "C" void copyArrayToDevice(cl_mem memObj, const void* hostPtr, size_t offset, size_t size);
extern "C" void setParameters(sim_params * m_params);
extern "C" void setParametersHost(sim_params * host_params);
extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue);
extern "C" void integrateSystem(cl_mem d_Pos, cl_mem d_Vel, float deltaTime, uint numParticles);
extern "C" void calcHash(cl_mem d_Hash, cl_mem d_Index, cl_mem d_Pos, int numParticles);
static void memsetOCL(cl_mem d_Data, uint val, uint N);
extern "C" void findCellBoundsAndReorder(cl_mem d_CellStart, cl_mem d_CellEnd, cl_mem d_ReorderedPos,
    cl_mem d_ReorderedVel, cl_mem d_Hash, cl_mem d_Index, cl_mem d_Pos, cl_mem d_Vel,
    uint numParticles, uint numCells);
extern "C" void collide(cl_mem d_Vel, cl_mem d_ReorderedPos, cl_mem d_ReorderedVel, cl_mem d_Index,
    cl_mem d_CellStart, cl_mem d_CellEnd, uint   numParticles, uint   numCells);
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
