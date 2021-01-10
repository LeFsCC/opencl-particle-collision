#pragma once
#ifndef SPHERESYSTEM_H
#define SPHERESYSTEM_H

#include <cl/cl.h>

typedef unsigned int uint;
#define check_error(a, b) __oclCheckErrorEX(a, b, 0, __FILE__ , __LINE__)
static size_t wgSize = 64;

struct float3 {
    float x, y, z;
};

struct uint3 {
    unsigned int x, y, z;
};

typedef struct {
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;
} sim_params;

static inline  float3 make_float3(float x, float y, float z) {
    float3 t; t.x = x; t.y = y; t.z = z; return t;
}
 
class Spheres {

public:
    enum ParticleArray {
        POSITION,
        VELOCITY,
    };
	Spheres(uint, uint3);
    void init_params();
    void init_data();
    float* get_array(ParticleArray array);
    void set_array(ParticleArray array, const float* data, int start, int count);
    void reset();
    void update(float deltaTime);
    float* get_pos() { return cpu_pos; }

protected:
    uint num_particles;

    float* cpu_pos;
    float* cpu_vel;
    float* cpu_reco_pos;
    float* cpu_reco_vel;
    uint* cpu_cell_start;
    uint* cpu_cell_end;
    uint* cpu_hash;
    uint* cpu_index;

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