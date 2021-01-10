#include "utils.h"
#include <iostream>
#include <fstream>

cl_context context_cl;
cl_command_queue cmdq;
static cl_mem params;
static cl_program particle_clprogram, sort_clprogram;
static cl_command_queue default_cmdq;


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

extern "C" void prepare_ocl_platform() {
    cl_uint counts;
    cl_platform_id* cpPlatform;
    cl_device_id cdDevices;
    cl_int error_code;

    error_code = clGetPlatformIDs(0, 0, &counts);
    check_error(error_code, 0);

    cpPlatform =  new cl_platform_id[counts];
    error_code = clGetPlatformIDs(counts, cpPlatform, NULL);
    check_error(error_code, 0);

    error_code = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, &cdDevices, NULL);
    check_error(error_code, 0);

    context_cl = clCreateContext(NULL, 1, &cdDevices, 0, 0, &error_code);
    check_error(error_code, 0);

    cmdq = clCreateCommandQueue(context_cl, cdDevices, 0, &error_code);
    check_error(error_code, 0);

    size_t kernelLength;
    char* cParticles = clcode("particle.cl", &kernelLength);
    check_error(cParticles != NULL, 1);

    particle_clprogram = clCreateProgramWithSource(context_cl, 1, (const char**)&cParticles, NULL, &error_code);
    check_error(error_code, 0);

    error_code = clBuildProgram(particle_clprogram, 0, 0, NULL, NULL, NULL);
    check_error(error_code, 0);

    ckIntegrate = clCreateKernel(particle_clprogram, "integrate", &error_code);
    check_error(error_code, 0);

    ckCalcHash = clCreateKernel(particle_clprogram, "calcHash", &error_code);
    check_error(error_code, 0);

    ckMemset = clCreateKernel(particle_clprogram, "Memset", &error_code);
    check_error(error_code, 0);

    ckFindCellBoundsAndReorder = clCreateKernel(particle_clprogram, "findCellBoundsAndReorder", &error_code);
    check_error(error_code, 0);

    ckCollide = clCreateKernel(particle_clprogram, "collide", &error_code);
    check_error(error_code, 0);

    create_gpu_buffer(&params, sizeof(sim_params));
    default_cmdq = cmdq;

    char* cBitonicSort = clcode("sort.cl", &kernelLength);
    sort_clprogram = clCreateProgramWithSource(context_cl, 1, (const char**)&cBitonicSort, &kernelLength, &error_code);
    check_error(error_code, 0);

    error_code = clBuildProgram(sort_clprogram, 0, NULL, NULL, NULL, NULL);
    check_error(error_code, 0);

    ckBitonicSortLocal1 = clCreateKernel(sort_clprogram, "bitonicSortLocal1", &error_code);
    check_error(error_code, 0);

    ckBitonicMergeGlobal = clCreateKernel(sort_clprogram, "bitonicMergeGlobal", &error_code);
    check_error(error_code, 0);

    default_cmdq = cmdq;

    free(cParticles);
    free(cBitonicSort);
}

extern "C" void create_gpu_buffer(cl_mem * memObj, size_t size) {
    cl_int error_code;
    *memObj = clCreateBuffer(context_cl, CL_MEM_READ_WRITE, size, NULL, &error_code);
    check_error(error_code, 0);
}

extern "C" void release_gpu_buffer(cl_mem memObj) {
    cl_int error_code;
    error_code = clReleaseMemObject(memObj);
    check_error(error_code, 0);
}

extern "C" void get_data_from_gpu(void* hostPtr, cl_mem memObj, size_t size) {
    cl_int error_code;
    error_code = clEnqueueReadBuffer(cmdq, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    check_error(error_code, 0);
}

extern "C" void set_data_on_gpu(cl_mem memObj, const void* hostPtr, size_t offset, size_t size) {
    cl_int error_code;
    error_code = clEnqueueWriteBuffer(cmdq, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    check_error(error_code, 0);
}

extern "C" void set_constants(sim_params * m_params) {
    set_data_on_gpu(params, m_params, 0, sizeof(sim_params));
}
 
extern"C" void merge_sort(cl_command_queue cqCommandQueue,cl_mem d_DstKey, cl_mem d_DstVal,cl_mem d_SrcKey,cl_mem d_SrcVal, 
    uint batch, uint arrayLength, uint dir) {

    if (arrayLength < 2)
        return;
    check_error(cl_uint(arrayLength & (arrayLength - 1)) == 1, 0);

    if (!cqCommandQueue)
        cqCommandQueue = default_cmdq;

    dir = (dir != 0);

    cl_int error_code;
    size_t localWorkSize, globalWorkSize;
    error_code = clSetKernelArg(ckBitonicSortLocal1, 0, sizeof(cl_mem), (void*)&d_DstKey);
    error_code |= clSetKernelArg(ckBitonicSortLocal1, 1, sizeof(cl_mem), (void*)&d_DstVal);
    error_code |= clSetKernelArg(ckBitonicSortLocal1, 2, sizeof(cl_mem), (void*)&d_SrcKey);
    error_code |= clSetKernelArg(ckBitonicSortLocal1, 3, sizeof(cl_mem), (void*)&d_SrcVal);
    check_error(error_code, 0);

    localWorkSize = LOCAL_SIZE_LIMIT / 2;
    globalWorkSize = batch * arrayLength / 2;
    error_code = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicSortLocal1, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    check_error(error_code, 0);

    for (unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
    {
        for (unsigned stride = size / 2; stride > 0; stride >>= 1)
        {
            if (stride >= LOCAL_SIZE_LIMIT)
            {
                //Launch merge
                error_code = clSetKernelArg(ckBitonicMergeGlobal, 0, sizeof(cl_mem), (void*)&d_DstKey);
                error_code |= clSetKernelArg(ckBitonicMergeGlobal, 1, sizeof(cl_mem), (void*)&d_DstVal);
                error_code |= clSetKernelArg(ckBitonicMergeGlobal, 2, sizeof(cl_mem), (void*)&d_DstKey);
                error_code |= clSetKernelArg(ckBitonicMergeGlobal, 3, sizeof(cl_mem), (void*)&d_DstVal);
                error_code |= clSetKernelArg(ckBitonicMergeGlobal, 4, sizeof(cl_uint), (void*)&arrayLength);
                error_code |= clSetKernelArg(ckBitonicMergeGlobal, 5, sizeof(cl_uint), (void*)&size);
                error_code |= clSetKernelArg(ckBitonicMergeGlobal, 6, sizeof(cl_uint), (void*)&stride);
                error_code |= clSetKernelArg(ckBitonicMergeGlobal, 7, sizeof(cl_uint), (void*)&dir);
                check_error(error_code, 0);

                localWorkSize = LOCAL_SIZE_LIMIT / 4;
                globalWorkSize = batch * arrayLength / 2;

                error_code = clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicMergeGlobal, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
                check_error(error_code, 0);
            }
        }
    }
}

static size_t uSnap(size_t a, size_t b) {
    return ((a % b) == 0) ? a : (a - (a % b) + b);
}

extern "C" void refresh_particles(cl_mem d_Pos, cl_mem d_Vel, float deltaTime, uint numParticles) {
    cl_int error_code;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    error_code = clSetKernelArg(ckIntegrate, 0, sizeof(cl_mem), (void*)&d_Pos);
    check_error(error_code, 0);

    error_code |= clSetKernelArg(ckIntegrate, 1, sizeof(cl_mem), (void*)&d_Vel);
    check_error(error_code, 0);
    error_code |= clSetKernelArg(ckIntegrate, 2, sizeof(cl_mem), (void*)&params);
    check_error(error_code, 0);
    error_code |= clSetKernelArg(ckIntegrate, 3, sizeof(float), (void*)&deltaTime);
    check_error(error_code, 0);
    error_code |= clSetKernelArg(ckIntegrate, 4, sizeof(uint), (void*)&numParticles);
    check_error(error_code, 0);

    error_code = clEnqueueNDRangeKernel(default_cmdq, ckIntegrate, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    check_error(error_code, 0);
}

extern "C" void reckon_hash(cl_mem d_Hash, cl_mem d_Index, cl_mem d_Pos, int numParticles) {
    cl_int error_code;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    error_code = clSetKernelArg(ckCalcHash, 0, sizeof(cl_mem), (void*)&d_Hash);
    error_code |= clSetKernelArg(ckCalcHash, 1, sizeof(cl_mem), (void*)&d_Index);
    error_code |= clSetKernelArg(ckCalcHash, 2, sizeof(cl_mem), (void*)&d_Pos);
    error_code |= clSetKernelArg(ckCalcHash, 3, sizeof(cl_mem), (void*)&params);
    error_code |= clSetKernelArg(ckCalcHash, 4, sizeof(uint), (void*)&numParticles);
    check_error(error_code, 0);

    error_code = clEnqueueNDRangeKernel(default_cmdq, ckCalcHash, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    check_error(error_code, 0);
}

static void memset_gpu(cl_mem d_Data, uint val, uint N) {
    cl_int error_code;
    size_t globalWorkSize = uSnap(N, wgSize);

    error_code = clSetKernelArg(ckMemset, 0, sizeof(cl_mem), (void*)&d_Data);
    error_code |= clSetKernelArg(ckMemset, 1, sizeof(cl_uint), (void*)&val);
    error_code |= clSetKernelArg(ckMemset, 2, sizeof(cl_uint), (void*)&N);
    check_error(error_code, 0);

    error_code = clEnqueueNDRangeKernel(default_cmdq, ckMemset, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    check_error(error_code, 0);
}

extern "C" void find_cell_bounds_and_reorder(cl_mem d_CellStart, cl_mem d_CellEnd, cl_mem d_ReorderedPos,
    cl_mem d_ReorderedVel, cl_mem d_Hash, cl_mem d_Index, cl_mem d_Pos, cl_mem d_Vel,
    uint numParticles, uint numCells) {
    cl_int error_code;
    memset_gpu(d_CellStart, 0xFFFFFFFFU, numCells);
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    error_code = clSetKernelArg(ckFindCellBoundsAndReorder, 0, sizeof(cl_mem), (void*)&d_CellStart);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 1, sizeof(cl_mem), (void*)&d_CellEnd);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 2, sizeof(cl_mem), (void*)&d_ReorderedPos);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 3, sizeof(cl_mem), (void*)&d_ReorderedVel);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 4, sizeof(cl_mem), (void*)&d_Hash);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 5, sizeof(cl_mem), (void*)&d_Index);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 6, sizeof(cl_mem), (void*)&d_Pos);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 7, sizeof(cl_mem), (void*)&d_Vel);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 8, (wgSize + 1) * sizeof(cl_uint), NULL);
    error_code |= clSetKernelArg(ckFindCellBoundsAndReorder, 9, sizeof(cl_uint), (void*)&numParticles);
    check_error(error_code, 0);

    error_code = clEnqueueNDRangeKernel(default_cmdq, ckFindCellBoundsAndReorder, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    check_error(error_code, 0);
}

extern "C" void collide(cl_mem d_Vel, cl_mem d_ReorderedPos, cl_mem d_ReorderedVel, cl_mem d_Index,
    cl_mem d_CellStart, cl_mem d_CellEnd, uint   numParticles, uint   numCells) {
    cl_int error_code;
    size_t globalWorkSize = uSnap(numParticles, wgSize);

    error_code = clSetKernelArg(ckCollide, 0, sizeof(cl_mem), (void*)&d_Vel);
    error_code |= clSetKernelArg(ckCollide, 1, sizeof(cl_mem), (void*)&d_ReorderedPos);
    error_code |= clSetKernelArg(ckCollide, 2, sizeof(cl_mem), (void*)&d_ReorderedVel);
    error_code |= clSetKernelArg(ckCollide, 3, sizeof(cl_mem), (void*)&d_Index);
    error_code |= clSetKernelArg(ckCollide, 4, sizeof(cl_mem), (void*)&d_CellStart);
    error_code |= clSetKernelArg(ckCollide, 5, sizeof(cl_mem), (void*)&d_CellEnd);
    error_code |= clSetKernelArg(ckCollide, 6, sizeof(cl_mem), (void*)&params);
    error_code |= clSetKernelArg(ckCollide, 7, sizeof(uint), (void*)&numParticles);
    check_error(error_code, 0);

    error_code = clEnqueueNDRangeKernel(default_cmdq, ckCollide, 1, NULL, &globalWorkSize, &wgSize, 0, NULL, NULL);
    check_error(error_code, 0);
}

