#include "utils.h"
#include <iostream>
#include <fstream>

cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
static cl_mem params;
static simParams_t h_params;
static cl_program cpParticles;

static cl_kernel ckIntegrate, ckCalcHash, ckMemset, ckFindCellBoundsAndReorder, ckCollide;
static cl_command_queue cqDefaultCommandQue;



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
    if (ciErrNum != CL_SUCCESS) {
        std::cout << "error build" << std::endl;
    }

    ckIntegrate = clCreateKernel(cpParticles, "integrate", &ciErrNum);

    oclCheckError(ciErrNum, CL_SUCCESS);
    CL_INVALID_ARG_VALUE;

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
}

extern "C" void shutdownOpenCL(void) {
    cl_int ciErrNum;
    //closeParticles();
    //closeBitonicSort();
    ciErrNum = clReleaseCommandQueue(cqCommandQueue);
    ciErrNum |= clReleaseContext(cxGPUContext);
    oclCheckError(ciErrNum, CL_SUCCESS);
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

extern "C" void copyArrayFromDevice(void* hostPtr, memHandle_t memObj, unsigned int vbo, size_t size) {
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