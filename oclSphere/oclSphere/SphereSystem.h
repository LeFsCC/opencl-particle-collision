
/*
    关于粒子系统的参数初始化
    cpu 和 gpu 之间的数据(位置)交换
*/

#pragma once
#ifndef SPHERESYSTEM_H
#define SPHERESYSTEM_H

#include <cl/cl.h>

typedef unsigned int uint;
static size_t world_size = 64;

struct float3 {
    float x, y, z;
};

struct uint3 {
    unsigned int x, y, z;
};

// 物理场景的参数列表
typedef struct {
    float3 gravity;        // 重力
    float global_damping;  // 全局反弹系数
    uint3 grid_size;       // 网格尺寸
    uint num_cells;        // 细胞个数
    float3 world_origin;   // 世界原点
    float3 cell_size;      // 细胞大小
    float spring;          // 弹性系数
    float damping;         // 阻尼系数
    float shear;           // 剪应力系数
    float attraction;      // 引力
    float boundaryDamping; // 边界反弹系数
} sim_params;

// 将三个float存在一个结构体中
static inline  float3 get_float_array(float x, float y, float z) {
    float3 t; 
    t.x = x; 
    t.y = y; 
    t.z = z; 
    return t;
}

class Spheres {

public:
    enum attr_type { POS, VEL};
	Spheres(uint, uint3);
    void update(float delta_time);   // 刷新函数
    float* get_pos() { return cpu_pos; }   // 得到粒子的位置
    void init_particle_params();    // 初始化粒子系统

private:
    void init_params();      // 初始化物理引擎参数
    void init_buffer();      // 初始化小球的半径，位置等参数，并存在内存和gpu里
    float* get_gpu_data(attr_type array);   // 从gpu中拿到数据
    void set_gpu_data(attr_type array, const float* data, int start, int count);  // 将数据存在gpu里

    uint num_particles;
    float* cpu_pos;
    float* cpu_vel;

    cl_mem          gpu_pos;   // 位置信息
    cl_mem          gpu_vel;   // 速度信息
    cl_mem     gpu_reod_pos;   // 排序以后的位置
    cl_mem     gpu_reod_vel;   // 排序以后的速度
    cl_mem         gpu_hash;   // 哈希值
    cl_mem        gpu_index;   // 索引值
    cl_mem   gpu_cell_start;   // 细胞开始地址
    cl_mem     gpu_cell_end;   // 细胞结束地址

    sim_params params;         // gpu需要拿到的物理引擎参数
    uint3 grid_size;           // 格子大小
    uint num_grid_cells;       // 细胞个数
};

#endif SPHERESYSTEM_H