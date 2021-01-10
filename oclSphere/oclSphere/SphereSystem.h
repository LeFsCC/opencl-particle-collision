#pragma once
#ifndef SPHERESYSTEM_H
#define SPHERESYSTEM_H

#include <cl/cl.h>

typedef unsigned int uint;
#define check_error(a, b) ocl_check_error(a, b, 0, __FILE__ , __LINE__)
static size_t world_size = 64;

struct float3 {
    float x, y, z;
};

struct uint3 {
    unsigned int x, y, z;
};

typedef struct {
    float3 gravity;
    float global_damping;
    uint3 grid_size;
    uint num_cells;
    float3 world_origin;
    float3 cell_size;
    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} sim_params;

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
    void init_params();
    void init_buffer();
    float* get_gpu_data(attr_type array);
    void set_gpu_data(attr_type array, const float* data, int start, int count);
    void init_particle_params();
    void update(float deltaTime);
    float* get_pos() { return cpu_pos; }

protected:
    uint num_particles;

    float* cpu_pos;
    float* cpu_vel;

    cl_mem          gpu_pos;
    cl_mem          gpu_vel;
    cl_mem     gpu_reco_pos;
    cl_mem     gpu_reco_vel;
    cl_mem         gpu_hash;
    cl_mem        gpu_index;
    cl_mem   gpu_cell_start;
    cl_mem     gpu_cell_end;

    sim_params params;
    uint3 grid_size;
    uint num_grid_cells;
};

#endif SPHERESYSTEM_H