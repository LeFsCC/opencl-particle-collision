#include "utils.h"
#include <iostream>
#include <fstream>


cl_context context;
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
    context = clCreateContext(NULL, 1, &cdDevices, 0, 0, &error_code);
    check_error(error_code, 0);
    cmdq = clCreateCommandQueue(context, cdDevices, 0, &error_code);
    check_error(error_code, 0);
    size_t kernelLength;
    char* cParticles = clcode("particle.cl", &kernelLength);
    check_error(cParticles != NULL, 1);
    particle_clprogram = clCreateProgramWithSource(context, 1, (const char**)&cParticles, NULL, &error_code);
    check_error(error_code, 0);
    error_code = clBuildProgram(particle_clprogram, 0, 0, NULL, NULL, NULL);
    check_error(error_code, 0);
    init_system_ck = clCreateKernel(particle_clprogram, "integrate", &error_code);
    check_error(error_code, 0);
    calc_hash_ck = clCreateKernel(particle_clprogram, "calc_hash", &error_code);
    check_error(error_code, 0);
    memset_ck = clCreateKernel(particle_clprogram, "memset_gpu", &error_code);
    check_error(error_code, 0);
    find_bounds_and_reorder_ck = clCreateKernel(particle_clprogram, "find_cell_bounds_reorder", &error_code);
    check_error(error_code, 0);
    collide_ck = clCreateKernel(particle_clprogram, "collide", &error_code);
    check_error(error_code, 0);
    create_gpu_buffer(&params, sizeof(sim_params));
    default_cmdq = cmdq;
    char* cBitonicSort = clcode("sort.cl", &kernelLength);
    sort_clprogram = clCreateProgramWithSource(context, 1, (const char**)&cBitonicSort, &kernelLength, &error_code);
    check_error(error_code, 0);
    error_code = clBuildProgram(sort_clprogram, 0, NULL, NULL, NULL, NULL);
    check_error(error_code, 0);
    gpu_sort_ck = clCreateKernel(sort_clprogram, "merge_sort", &error_code);
    check_error(error_code, 0);
    gpu_sort_merge_ck = clCreateKernel(sort_clprogram, "merge", &error_code);
    check_error(error_code, 0);
    default_cmdq = cmdq;
    free(cParticles);
    free(cBitonicSort);
}

extern "C" void create_gpu_buffer(cl_mem * memObj, size_t size) {
    cl_int error_code;
    *memObj = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &error_code);
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
 
extern"C" void merge_sort(cl_command_queue cmdq,cl_mem dst_key, cl_mem dst_val,cl_mem src_key,cl_mem src_val, 
    uint batch, uint arr_len, uint dir) {

    if (arr_len < 2)
        return;
    check_error(cl_uint(arr_len & (arr_len - 1)) == 1, 0);

    if (!cmdq)
        cmdq = default_cmdq;

    dir = (dir != 0);

    cl_int error_code;
    size_t local_work_size, global_work_size;
    error_code = clSetKernelArg(gpu_sort_ck, 0, sizeof(cl_mem), (void*)&dst_key);
    check_error(error_code, 0);
    error_code = clSetKernelArg(gpu_sort_ck, 1, sizeof(cl_mem), (void*)&dst_val);
    check_error(error_code, 0);
    error_code = clSetKernelArg(gpu_sort_ck, 2, sizeof(cl_mem), (void*)&src_key);
    check_error(error_code, 0);
    error_code = clSetKernelArg(gpu_sort_ck, 3, sizeof(cl_mem), (void*)&src_val);
    check_error(error_code, 0);

    local_work_size = LIMIT / 2;
    global_work_size = batch * arr_len / 2;
    error_code = clEnqueueNDRangeKernel(cmdq, gpu_sort_ck, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    check_error(error_code, 0);

    for (unsigned int size = 2 * LIMIT; size <= arr_len; size <<= 1) {
        for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
            if (stride >= LIMIT) {
                error_code = clSetKernelArg(gpu_sort_merge_ck, 0, sizeof(cl_mem), (void*)&dst_key);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 1, sizeof(cl_mem), (void*)&dst_val);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 2, sizeof(cl_mem), (void*)&dst_key);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 3, sizeof(cl_mem), (void*)&dst_val);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 4, sizeof(cl_uint), (void*)&arr_len);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 5, sizeof(cl_uint), (void*)&size);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 6, sizeof(cl_uint), (void*)&stride);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 7, sizeof(cl_uint), (void*)&dir);
                check_error(error_code, 0);

                local_work_size = LIMIT / 4;
                global_work_size = batch * arr_len / 2;

                error_code = clEnqueueNDRangeKernel(cmdq, gpu_sort_merge_ck, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
                check_error(error_code, 0);
            }
        }
    }
}

extern "C" void refresh_particles(cl_mem gpu_pos, cl_mem gpu_vel, float delta_time, uint num_particles) {
    cl_int error_code;
    size_t globalWorkSize = get_exact_div_par(num_particles, world_size);

    error_code = clSetKernelArg(init_system_ck, 0, sizeof(cl_mem), (void*)&gpu_pos);
    check_error(error_code, 0);
    error_code = clSetKernelArg(init_system_ck, 1, sizeof(cl_mem), (void*)&gpu_vel);
    check_error(error_code, 0);
    error_code = clSetKernelArg(init_system_ck, 2, sizeof(cl_mem), (void*)&params);
    check_error(error_code, 0);
    error_code = clSetKernelArg(init_system_ck, 3, sizeof(float), (void*)&delta_time);
    check_error(error_code, 0);
    error_code = clSetKernelArg(init_system_ck, 4, sizeof(uint), (void*)&num_particles);
    check_error(error_code, 0);
    error_code = clEnqueueNDRangeKernel(default_cmdq, init_system_ck, 1, NULL, &globalWorkSize, &world_size, 0, NULL, NULL);
    check_error(error_code, 0);
}

extern "C" void reckon_hash(cl_mem gpu_hash, cl_mem gpu_index, cl_mem gpu_pos, int num_particles) {
    cl_int error_code;
    size_t globalWorkSize = get_exact_div_par(num_particles, world_size);

    error_code = clSetKernelArg(calc_hash_ck, 0, sizeof(cl_mem), (void*)&gpu_hash);
    check_error(error_code, 0);
    error_code = clSetKernelArg(calc_hash_ck, 1, sizeof(cl_mem), (void*)&gpu_index);
    check_error(error_code, 0);
    error_code = clSetKernelArg(calc_hash_ck, 2, sizeof(cl_mem), (void*)&gpu_pos);
    check_error(error_code, 0);
    error_code = clSetKernelArg(calc_hash_ck, 3, sizeof(cl_mem), (void*)&params);
    check_error(error_code, 0);
    error_code = clSetKernelArg(calc_hash_ck, 4, sizeof(uint), (void*)&num_particles);
    check_error(error_code, 0);
    error_code = clEnqueueNDRangeKernel(default_cmdq, calc_hash_ck, 1, NULL, &globalWorkSize, &world_size, 0, NULL, NULL);
    check_error(error_code, 0);
}

static void memset_gpu(cl_mem gpu_data, uint val, uint N) {
    cl_int error_code;
    size_t globalWorkSize = get_exact_div_par(N, world_size);
    error_code = clSetKernelArg(memset_ck, 0, sizeof(cl_mem), (void*)&gpu_data);
    check_error(error_code, 0);
    error_code = clSetKernelArg(memset_ck, 1, sizeof(cl_uint), (void*)&val);
    check_error(error_code, 0);
    error_code = clSetKernelArg(memset_ck, 2, sizeof(cl_uint), (void*)&N);
    check_error(error_code, 0);
    error_code = clEnqueueNDRangeKernel(default_cmdq, memset_ck, 1, NULL, &globalWorkSize, &world_size, 0, NULL, NULL);
    check_error(error_code, 0);
}

extern "C" void find_cell_bounds_and_reorder(cl_mem gpu_cell_start, cl_mem gpu_cell_end, cl_mem gpu_reordered_pos,
    cl_mem gpu_reordered_vel, cl_mem gpu_hash, cl_mem gpu_index, cl_mem gpu_pos, cl_mem gpu_vel,
    uint gpu_num_particles, uint gpu_num_cells) {
    cl_int error_code;
    memset_gpu(gpu_cell_start, 0xFFFFFFFFU, gpu_num_cells);
    size_t globalWorkSize = get_exact_div_par(gpu_num_particles, world_size);

    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 0, sizeof(cl_mem), (void*)&gpu_cell_start);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 1, sizeof(cl_mem), (void*)&gpu_cell_end);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 2, sizeof(cl_mem), (void*)&gpu_reordered_pos);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 3, sizeof(cl_mem), (void*)&gpu_reordered_vel);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 4, sizeof(cl_mem), (void*)&gpu_hash);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 5, sizeof(cl_mem), (void*)&gpu_index);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 6, sizeof(cl_mem), (void*)&gpu_pos);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 7, sizeof(cl_mem), (void*)&gpu_vel);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 8, (world_size + 1) * sizeof(cl_uint), NULL);
    check_error(error_code, 0);
    error_code = clSetKernelArg(find_bounds_and_reorder_ck, 9, sizeof(cl_uint), (void*)&gpu_num_particles);
    check_error(error_code, 0);
    error_code = clEnqueueNDRangeKernel(default_cmdq, find_bounds_and_reorder_ck, 1, NULL, &globalWorkSize, &world_size, 0, NULL, NULL);
    check_error(error_code, 0);
}

extern "C" void collide(cl_mem gpu_vel, cl_mem gpu_reordered_pos, cl_mem gpu_reordered_vel, cl_mem gpu_index,
    cl_mem gpu_cell_start, cl_mem gpu_cell_end, uint   num_particles, uint   num_cells) {
    cl_int error_code;
    size_t globalWorkSize = get_exact_div_par(num_particles, world_size);

    error_code = clSetKernelArg(collide_ck, 0, sizeof(cl_mem), (void*)&gpu_vel);
    check_error(error_code, 0);
    error_code = clSetKernelArg(collide_ck, 1, sizeof(cl_mem), (void*)&gpu_reordered_pos);
    check_error(error_code, 0);
    error_code = clSetKernelArg(collide_ck, 2, sizeof(cl_mem), (void*)&gpu_reordered_vel);
    check_error(error_code, 0);
    error_code = clSetKernelArg(collide_ck, 3, sizeof(cl_mem), (void*)&gpu_index);
    check_error(error_code, 0);
    error_code = clSetKernelArg(collide_ck, 4, sizeof(cl_mem), (void*)&gpu_cell_start);
    check_error(error_code, 0);
    error_code = clSetKernelArg(collide_ck, 5, sizeof(cl_mem), (void*)&gpu_cell_end);
    check_error(error_code, 0);
    error_code = clSetKernelArg(collide_ck, 6, sizeof(cl_mem), (void*)&params);
    check_error(error_code, 0);
    error_code = clSetKernelArg(collide_ck, 7, sizeof(uint), (void*)&num_particles);
    check_error(error_code, 0);
    error_code = clEnqueueNDRangeKernel(default_cmdq, collide_ck, 1, NULL, &globalWorkSize, &world_size, 0, NULL, NULL);
    check_error(error_code, 0);
}

