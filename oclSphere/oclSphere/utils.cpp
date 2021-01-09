#include "utils.h"
#include <iostream>
#include <fstream>

cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
static cl_mem params;
static simParams_t h_params;
static cl_program cpParticles;


static cl_command_queue cqDefaultCommandQue;

static cl_kernel
ckBitonicSortLocal,
ckBitonicSortLocal1,
ckBitonicMergeGlobal,
ckBitonicMergeLocal;

static cl_program cpBitonicSort;


char* clcode(const char* fileName, size_t* length) {
    FILE* fp = fopen(fileName, "rb");
    fseek(fp, 0, SEEK_END);
    size_t fileset = ftell(fp);
    *length = fileset;
    fseek(fp, 0, SEEK_SET);
    char* clcode = new char[fileset + 1];
    fread(clcode, 1, fileset, fp);
    clcode[fileset] = 0;
    fclose(fp);
    //printf("%s\n", clcode);
    return clcode;
}

extern "C" void startupOpenCL() {
    cl_uint counts;
    cl_platform_id* cpPlatform;
    cl_device_id cdDevices;
    cl_int ciErrNum;

    ciErrNum = clGetPlatformIDs(0, 0, &counts);
    oclCheckError(ciErrNum, CL_SUCCESS);

    cpPlatform =  new cl_platform_id[counts];
    ciErrNum = clGetPlatformIDs(counts, cpPlatform, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
    for (int i = 0; i < counts; i++) {
		size_t size;
        ciErrNum = clGetPlatformInfo(cpPlatform[i], CL_PLATFORM_NAME, 0, NULL, &size);
		char* name = new char[size];
        ciErrNum = clGetPlatformInfo(cpPlatform[i], CL_PLATFORM_NAME, size, name, NULL);
		std::cout << "GPU" << i + 1 << "Ãû³Æ£º" << name << std::endl;
		delete name;
	}


    ciErrNum = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, &cdDevices, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Create the context
    cxGPUContext = clCreateContext(NULL, 1, &cdDevices, 0, 0, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    //Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices, 0, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    //initBitonicSort(cxGPUContext, cqCommandQueue, argv);

    size_t kernelLength;
    char* cParticles = clcode("particle.cl", &kernelLength);
    oclCheckError(cParticles != NULL, 1);

    cpParticles = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&cParticles, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clBuildProgram(cpParticles, 0, 0, NULL, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ckIntegrate = clCreateKernel(cpParticles, "integrate", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ckCalcHash = clCreateKernel(cpParticles, "calcHash", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ckMemset = clCreateKernel(cpParticles, "Memset", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ckFindCellBoundsAndReorder = clCreateKernel(cpParticles, "findCellBoundsAndReorder", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ckCollide = clCreateKernel(cpParticles, "collide", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    allocateArray(&params, sizeof(simParams_t));
    cqDefaultCommandQue = cqCommandQueue;

    free(cParticles);
    initBitonicSort(cxGPUContext, cqCommandQueue);
}

extern "C" void allocateArray(memHandle_t * memObj, size_t size) {
    cl_int ciErrNum;
    *memObj = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void freeArray(memHandle_t memObj) {
    cl_int ciErrNum;
    ciErrNum = clReleaseMemObject(memObj);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void copyArrayFromDevice(void* hostPtr, memHandle_t memObj, size_t size) {
    cl_int ciErrNum;
    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void copyArrayToDevice(memHandle_t memObj, const void* hostPtr, size_t offset, size_t size) {
    cl_int ciErrNum;
    ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void setParameters(simParams_t * m_params) {
    copyArrayToDevice(params, m_params, 0, sizeof(simParams_t));
}

extern "C" void setParametersHost(simParams_t * host_params) {
    memcpy(&h_params, host_params, sizeof(simParams_t));
}

extern "C" void closeParticles(void) {
    cl_int ciErrNum;
    ciErrNum = clReleaseMemObject(params);
    ciErrNum |= clReleaseKernel(ckCollide);
    ciErrNum |= clReleaseKernel(ckFindCellBoundsAndReorder);
    ciErrNum |= clReleaseKernel(ckMemset);
    ciErrNum |= clReleaseKernel(ckCalcHash);
    ciErrNum |= clReleaseKernel(ckIntegrate);
    oclCheckError(ciErrNum, CL_SUCCESS);
}
 
static cl_uint factorRadix2(cl_uint& log2L, cl_uint L) {
    if (!L) {
        log2L = 0;
        return 0;
    }
    else {
        for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
        return L;
    }
}
 
extern"C" void bitonicSort(
    cl_command_queue cqCommandQueue,
    cl_mem d_DstKey,
    cl_mem d_DstVal,
    cl_mem d_SrcKey,
    cl_mem d_SrcVal, 
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir
) {
    if (arrayLength < 2)
        return;

    //Only power-of-two array lengths are supported so far
    cl_uint log2L;

    cl_uint factorizationRemainder = arrayLength & (arrayLength - 1);
    //std::cout << factorizationRemainder << std::endl;
    oclCheckError(factorizationRemainder == 1, CL_SUCCESS);

    if (!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    dir = (dir != 0);

    cl_int ciErrNum;
    size_t localWorkSize, globalWorkSize;

    if (arrayLength <= LOCAL_SIZE_LIMIT)
    {
        oclCheckError((batch * arrayLength) % LOCAL_SIZE_LIMIT == 0, CL_SUCCESS);
        ciErrNum = clSetKernelArg(ckBitonicSortLocal, 0, sizeof(cl_mem), (void*)&d_DstKey);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 1, sizeof(cl_mem), (void*)&d_DstVal);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 2, sizeof(cl_mem), (void*)&d_SrcKey);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 3, sizeof(cl_mem), (void*)&d_SrcVal);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 4, sizeof(cl_uint), (void*)&arrayLength);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal, 5, sizeof(cl_uint), (void*)&dir);
        oclCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicSortLocal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);
    }
    else
    {
        //Launch bitonicSortLocal1
        ciErrNum = clSetKernelArg(ckBitonicSortLocal1, 0, sizeof(cl_mem), (void*)&d_DstKey);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal1, 1, sizeof(cl_mem), (void*)&d_DstVal);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal1, 2, sizeof(cl_mem), (void*)&d_SrcKey);
        ciErrNum |= clSetKernelArg(ckBitonicSortLocal1, 3, sizeof(cl_mem), (void*)&d_SrcVal);
        oclCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize = LOCAL_SIZE_LIMIT / 2;
        globalWorkSize = batch * arrayLength / 2;
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicSortLocal1, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        for (unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
        {
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
            {
                if (stride >= LOCAL_SIZE_LIMIT)
                {
                    //Launch bitonicMergeGlobal
                    ciErrNum = clSetKernelArg(ckBitonicMergeGlobal, 0, sizeof(cl_mem), (void*)&d_DstKey);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 1, sizeof(cl_mem), (void*)&d_DstVal);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 2, sizeof(cl_mem), (void*)&d_DstKey);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 3, sizeof(cl_mem), (void*)&d_DstVal);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 4, sizeof(cl_uint), (void*)&arrayLength);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 5, sizeof(cl_uint), (void*)&size);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 6, sizeof(cl_uint), (void*)&stride);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeGlobal, 7, sizeof(cl_uint), (void*)&dir);
                    oclCheckError(ciErrNum, CL_SUCCESS);

                    localWorkSize = LOCAL_SIZE_LIMIT / 4;
                    globalWorkSize = batch * arrayLength / 2;

                    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicMergeGlobal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
                    oclCheckError(ciErrNum, CL_SUCCESS);
                }
                else
                {
                    //Launch bitonicMergeLocal
                    ciErrNum = clSetKernelArg(ckBitonicMergeLocal, 0, sizeof(cl_mem), (void*)&d_DstKey);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 1, sizeof(cl_mem), (void*)&d_DstVal);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 2, sizeof(cl_mem), (void*)&d_DstKey);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 3, sizeof(cl_mem), (void*)&d_DstVal);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 4, sizeof(cl_uint), (void*)&arrayLength);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 5, sizeof(cl_uint), (void*)&stride);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 6, sizeof(cl_uint), (void*)&size);
                    ciErrNum |= clSetKernelArg(ckBitonicMergeLocal, 7, sizeof(cl_uint), (void*)&dir);
                    oclCheckError(ciErrNum, CL_SUCCESS);

                    localWorkSize = LOCAL_SIZE_LIMIT / 2;
                    globalWorkSize = batch * arrayLength / 2;

                    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicMergeLocal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
                    oclCheckError(ciErrNum, CL_SUCCESS);
                    break;
                }
            }
        }
    }
}

extern "C" void initBitonicSort(cl_context cxGPUContext, cl_command_queue cqParamCommandQue) {
    cl_int ciErrNum;
    size_t kernelLength;

    char* cBitonicSort = clcode("BitonicSort_b.cl", &kernelLength);
    cpBitonicSort = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&cBitonicSort, &kernelLength, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clBuildProgram(cpBitonicSort, 0, NULL, NULL, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ckBitonicSortLocal = clCreateKernel(cpBitonicSort, "bitonicSortLocal", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    ckBitonicSortLocal1 = clCreateKernel(cpBitonicSort, "bitonicSortLocal1", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    ckBitonicMergeGlobal = clCreateKernel(cpBitonicSort, "bitonicMergeGlobal", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    ckBitonicMergeLocal = clCreateKernel(cpBitonicSort, "bitonicMergeLocal", &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    cqDefaultCommandQue = cqParamCommandQue;

    free(cBitonicSort);
}

static size_t uSnap(size_t a, size_t b) {
    return ((a % b) == 0) ? a : (a - (a % b) + b);
}

extern "C" void integrateSystem(memHandle_t d_Pos, memHandle_t d_Vel, float deltaTime, uint numParticles) {
    cl_int ciErrNum;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    ciErrNum = clSetKernelArg(ckIntegrate, 0, sizeof(cl_mem), (void*)&d_Pos);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum |= clSetKernelArg(ckIntegrate, 1, sizeof(cl_mem), (void*)&d_Vel);
    oclCheckError(ciErrNum, CL_SUCCESS);
    ciErrNum |= clSetKernelArg(ckIntegrate, 2, sizeof(cl_mem), (void*)&params);
    oclCheckError(ciErrNum, CL_SUCCESS);
    ciErrNum |= clSetKernelArg(ckIntegrate, 3, sizeof(float), (void*)&deltaTime);
    oclCheckError(ciErrNum, CL_SUCCESS);
    ciErrNum |= clSetKernelArg(ckIntegrate, 4, sizeof(uint), (void*)&numParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckIntegrate, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void calcHash(memHandle_t d_Hash, memHandle_t d_Index, memHandle_t d_Pos, int numParticles) {
    cl_int ciErrNum;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    ciErrNum = clSetKernelArg(ckCalcHash, 0, sizeof(cl_mem), (void*)&d_Hash);
    ciErrNum |= clSetKernelArg(ckCalcHash, 1, sizeof(cl_mem), (void*)&d_Index);
    ciErrNum |= clSetKernelArg(ckCalcHash, 2, sizeof(cl_mem), (void*)&d_Pos);
    ciErrNum |= clSetKernelArg(ckCalcHash, 3, sizeof(cl_mem), (void*)&params);
    ciErrNum |= clSetKernelArg(ckCalcHash, 4, sizeof(uint), (void*)&numParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckCalcHash, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

static void memsetOCL(memHandle_t d_Data, uint val, uint N) {
    cl_int ciErrNum;
    size_t globalWorkSize = uSnap(N, wgSize);

    ciErrNum = clSetKernelArg(ckMemset, 0, sizeof(cl_mem), (void*)&d_Data);
    ciErrNum |= clSetKernelArg(ckMemset, 1, sizeof(cl_uint), (void*)&val);
    ciErrNum |= clSetKernelArg(ckMemset, 2, sizeof(cl_uint), (void*)&N);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckMemset, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void findCellBoundsAndReorder(memHandle_t d_CellStart, memHandle_t d_CellEnd, memHandle_t d_ReorderedPos,
    memHandle_t d_ReorderedVel, memHandle_t d_Hash, memHandle_t d_Index, memHandle_t d_Pos, memHandle_t d_Vel,
    uint numParticles, uint numCells) {
    cl_int ciErrNum;
    memsetOCL(d_CellStart, 0xFFFFFFFFU, numCells);
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    ciErrNum = clSetKernelArg(ckFindCellBoundsAndReorder, 0, sizeof(cl_mem), (void*)&d_CellStart);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 1, sizeof(cl_mem), (void*)&d_CellEnd);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 2, sizeof(cl_mem), (void*)&d_ReorderedPos);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 3, sizeof(cl_mem), (void*)&d_ReorderedVel);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 4, sizeof(cl_mem), (void*)&d_Hash);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 5, sizeof(cl_mem), (void*)&d_Index);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 6, sizeof(cl_mem), (void*)&d_Pos);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 7, sizeof(cl_mem), (void*)&d_Vel);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 8, (wgSize + 1) * sizeof(cl_uint), NULL);
    ciErrNum |= clSetKernelArg(ckFindCellBoundsAndReorder, 9, sizeof(cl_uint), (void*)&numParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckFindCellBoundsAndReorder, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

extern "C" void collide(memHandle_t d_Vel, memHandle_t d_ReorderedPos, memHandle_t d_ReorderedVel, memHandle_t d_Index,
    memHandle_t d_CellStart, memHandle_t d_CellEnd, uint   numParticles, uint   numCells) {
    cl_int ciErrNum;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    ciErrNum = clSetKernelArg(ckCollide, 0, sizeof(cl_mem), (void*)&d_Vel);
    ciErrNum |= clSetKernelArg(ckCollide, 1, sizeof(cl_mem), (void*)&d_ReorderedPos);
    ciErrNum |= clSetKernelArg(ckCollide, 2, sizeof(cl_mem), (void*)&d_ReorderedVel);
    ciErrNum |= clSetKernelArg(ckCollide, 3, sizeof(cl_mem), (void*)&d_Index);
    ciErrNum |= clSetKernelArg(ckCollide, 4, sizeof(cl_mem), (void*)&d_CellStart);
    ciErrNum |= clSetKernelArg(ckCollide, 5, sizeof(cl_mem), (void*)&d_CellEnd);
    ciErrNum |= clSetKernelArg(ckCollide, 6, sizeof(cl_mem), (void*)&params);
    ciErrNum |= clSetKernelArg(ckCollide, 7, sizeof(uint), (void*)&numParticles);
    oclCheckError(ciErrNum, CL_SUCCESS);

    ciErrNum = clEnqueueNDRangeKernel(cqDefaultCommandQue, ckCollide, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    oclCheckError(ciErrNum, CL_SUCCESS);
}

