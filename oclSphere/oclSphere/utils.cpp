#include "utils.h"
#include <iostream>
#include <fstream>


cl_context context;
cl_command_queue cmdq;
static cl_mem params;
static cl_program particle_clprogram;
static cl_command_queue default_cmdq;

// 读文件
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

// 准备opencl环境
extern "C" void prepare_ocl_platform() {
    cl_uint counts;
    cl_platform_id* cpPlatform;
    cl_device_id cdDevices;
    cl_int error_code;

    // 获取平台
    error_code = clGetPlatformIDs(0, 0, &counts);
    check_error(error_code, 0);
    cpPlatform =  new cl_platform_id[counts];
    error_code = clGetPlatformIDs(counts, cpPlatform, NULL);
    check_error(error_code, 0);
    // 从平台中获取设备
    error_code = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, &cdDevices, NULL);
    check_error(error_code, 0);
    // 创建上下文
    context = clCreateContext(NULL, 1, &cdDevices, 0, 0, &error_code);
    check_error(error_code, 0);
    // 创建命令队列
    cmdq = clCreateCommandQueue(context, cdDevices, 0, &error_code);
    check_error(error_code, 0);
    // 读取cl文件, 创建程序
    size_t kernelLength;
    char* particle_cl_code = clcode("particle.cl", &kernelLength);
    check_error(particle_cl_code != NULL, 1);
    particle_clprogram = clCreateProgramWithSource(context, 1, (const char**)&particle_cl_code, NULL, &error_code);
    check_error(error_code, 0);
    // 编译程序
    error_code = clBuildProgram(particle_clprogram, 0, 0, NULL, NULL, NULL);
    check_error(error_code, 0);
    // 创建内核
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
    gpu_sort_ck = clCreateKernel(particle_clprogram, "merge_sort", &error_code);
    check_error(error_code, 0);
    gpu_sort_merge_ck = clCreateKernel(particle_clprogram, "merge", &error_code);
    check_error(error_code, 0);
    create_gpu_buffer(&params, sizeof(sim_params));
    default_cmdq = cmdq;
    free(particle_cl_code);
}

// 创建gpu buffer
extern "C" void create_gpu_buffer(cl_mem * memObj, size_t size) {
    cl_int error_code;
    *memObj = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &error_code);
    check_error(error_code, 0);
}

// 从gpu上获取数据
extern "C" void get_data_from_gpu(void* hostPtr, cl_mem memObj, size_t size) {
    cl_int error_code;
    error_code = clEnqueueReadBuffer(cmdq, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    check_error(error_code, 0);
}

// 将数据写在gpu上
extern "C" void set_data_on_gpu(cl_mem memObj, const void* hostPtr, size_t offset, size_t size) {
    cl_int error_code;
    error_code = clEnqueueWriteBuffer(cmdq, memObj, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL);
    check_error(error_code, 0);
}

// 设置常量
extern "C" void set_constants(sim_params * m_params) {
    set_data_on_gpu(params, m_params, 0, sizeof(sim_params));
}

// 归并排序
extern"C" void merge_sort(cl_mem dst_key, cl_mem dst_val,uint arr_len) {
    if (arr_len < 2 || cl_uint(arr_len & (arr_len - 1)) == 1)
        exit(0);

    uint dir = 1;
    cl_int error_code;
    size_t local_work_size, global_work_size;
    error_code = clSetKernelArg(gpu_sort_ck, 0, sizeof(cl_mem), (void*)&dst_key);
    check_error(error_code, 0);
    error_code = clSetKernelArg(gpu_sort_ck, 1, sizeof(cl_mem), (void*)&dst_val);
    check_error(error_code, 0);
    error_code = clSetKernelArg(gpu_sort_ck, 2, sizeof(cl_mem), (void*)&dst_key);
    check_error(error_code, 0);
    error_code = clSetKernelArg(gpu_sort_ck, 3, sizeof(cl_mem), (void*)&dst_val);
    check_error(error_code, 0);

    local_work_size = LIMIT / 4;
    global_work_size = arr_len / 2;
    // 执行内核
    error_code = clEnqueueNDRangeKernel(default_cmdq, gpu_sort_ck, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
    check_error(error_code, 0);

    for (uint i = 2 * LIMIT; i <= arr_len; i *= 2) {
        for (uint j = i / 2; j > 0; j /= 2) {
            if (j >= LIMIT) {
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
                error_code = clSetKernelArg(gpu_sort_merge_ck, 5, sizeof(cl_uint), (void*)&i);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 6, sizeof(cl_uint), (void*)&j);
                check_error(error_code, 0);
                error_code = clSetKernelArg(gpu_sort_merge_ck, 7, sizeof(cl_uint), (void*)&dir);
                check_error(error_code, 0);
                error_code = clEnqueueNDRangeKernel(default_cmdq, gpu_sort_merge_ck, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
                check_error(error_code, 0);
            }
        }
    }
}

// 刷新粒子
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

// 计算哈希
extern "C" void reckon_hash(cl_mem gpu_hash, cl_mem gpu_index, cl_mem gpu_pos, int num_particles) {
    cl_int error_code;
    size_t global_work_size = get_exact_div_par(num_particles, world_size);

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
    error_code = clEnqueueNDRangeKernel(default_cmdq, calc_hash_ck, 1, NULL, &global_work_size, &world_size, 0, NULL, NULL);
    check_error(error_code, 0);
}


// 计算细胞边界，并对速度和位置进行重排
extern "C" void find_cell_bounds_and_reorder(cl_mem gpu_cell_start, cl_mem gpu_cell_end, cl_mem gpu_reordered_pos,
    cl_mem gpu_reordered_vel, cl_mem gpu_hash, cl_mem gpu_index, cl_mem gpu_pos, cl_mem gpu_vel,
    uint gpu_num_particles, uint gpu_num_cells) {
    cl_int error_code;
    size_t global_work_size;

    // memset cell
    uint val = 0xFFFFFFFFU;
    global_work_size = get_exact_div_par(gpu_num_cells, world_size);
    error_code = clSetKernelArg(memset_ck, 0, sizeof(cl_mem), (void*)&gpu_cell_start);
    check_error(error_code, 0);
    error_code = clSetKernelArg(memset_ck, 1, sizeof(cl_uint), (void*)&val);
    check_error(error_code, 0);
    error_code = clSetKernelArg(memset_ck, 2, sizeof(cl_uint), (void*)&gpu_num_cells);
    check_error(error_code, 0);
    error_code = clEnqueueNDRangeKernel(default_cmdq, memset_ck, 1, NULL, &global_work_size, &world_size, 0, NULL, NULL);
    check_error(error_code, 0);

    // 找到边界并且进行重排序
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
    error_code = clEnqueueNDRangeKernel(default_cmdq, find_bounds_and_reorder_ck, 1, NULL, &global_work_size, &world_size, 0, NULL, NULL);
    check_error(error_code, 0);
}

// 碰撞检测
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

