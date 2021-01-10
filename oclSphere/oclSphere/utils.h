#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <cl/opencl.h>
#pragma warning( disable : 4996 )
#include <cl/cl.h>
#include "SphereSystem.h"
#include <fstream>
#define check_error(a, b) ocl_check_error(a, b, 0, __FILE__ , __LINE__)
typedef cl_mem cl_mem;
typedef unsigned int uint;
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a < b) ? a : b)
#define CLAMP(a, b, c) MIN(MAX(a, b), c)

static const uint LIMIT = 512U;

static cl_kernel init_system_ck, calc_hash_ck, memset_ck, find_bounds_and_reorder_ck, collide_ck;
static cl_kernel gpu_sort_ck, gpu_sort_merge_ck;
extern "C" void prepare_ocl_platform();
extern "C" void create_gpu_buffer(cl_mem * memObj, size_t size);
extern "C" void release_gpu_buffer(cl_mem memObj);
extern "C" void get_data_from_gpu(void* hostPtr, const cl_mem memObj, size_t size);
extern "C" void set_data_on_gpu(cl_mem memObj, const void* hostPtr, size_t offset, size_t size);
extern "C" void set_constants(sim_params * m_params);
extern "C" void refresh_particles(cl_mem d_Pos, cl_mem d_Vel, float deltaTime, uint numParticles);
extern "C" void reckon_hash(cl_mem d_Hash, cl_mem d_Index, cl_mem d_Pos, int numParticles);
static void memset_gpu(cl_mem d_Data, uint val, uint N);
extern "C" void find_cell_bounds_and_reorder(cl_mem d_CellStart, cl_mem d_CellEnd, cl_mem d_ReorderedPos,
    cl_mem d_ReorderedVel, cl_mem d_Hash, cl_mem d_Index, cl_mem d_Pos, cl_mem d_Vel,
    uint numParticles, uint numCells);
extern "C" void collide(cl_mem d_Vel, cl_mem d_ReorderedPos, cl_mem d_ReorderedVel, cl_mem d_Index,
    cl_mem d_CellStart, cl_mem d_CellEnd, uint   numParticles, uint   numCells);

extern"C" void merge_sort(cl_command_queue cqCommandQueue, cl_mem d_DstKey, cl_mem d_DstVal, cl_mem d_SrcKey,
    cl_mem d_SrcVal, unsigned int batch, unsigned int arrayLength, unsigned int dir);

static void ocl_check_error(cl_int iSample, cl_int iReference, void (*pCleanup)(int), const char* cFile, const int iLine) {
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

static size_t get_exact_div_par(size_t a, size_t b) {
    if (a % b == 0) {
        return a;
    }
    else {
        size_t c = a - a % b;
        return c + b;
    }
}
#endif UTILS_H
