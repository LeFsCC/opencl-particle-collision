#include "SphereSystem.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <cl/cl.h>
#include <GL/glew.h>
#include <iostream>

typedef cl_mem cl_mem;
typedef unsigned int uint;

static cl_command_queue default_cmdq;

static cl_mem params;

Spheres::Spheres(uint numParticles, uint3 gridSize) {
	num_particles = numParticles;
	cpu_pos = cpu_vel = 0;
	gpu_pos = gpu_vel = 0;
	grid_size = gridSize;
	init_params();
	init_buffer();
}

// 初始化物理引擎参数
void Spheres::init_params() {
	num_grid_cells = grid_size.x * grid_size.y * grid_size.z;
	params.grid_size = grid_size;
	params.num_cells = num_grid_cells;
	params.world_origin = get_float_array(-1.0f, -1.0f, -1.0f);
	params.cell_size = get_float_array(0.03f * 2.0f, 0.03f * 2.0f, 0.03f * 2.0f);
	params.spring = 0.2f;
	params.damping = 0.02f;
	params.shear = 0.01f;
	params.attraction = 0.0f;
	params.boundaryDamping = -0.5f;
	params.gravity = get_float_array(0.0f, -0.0003f, 0.0f);
	params.global_damping = 1.0f;
}

// 分配空间，并初始化为0
float* malset(float* addr, size_t sz) {
	addr = (float*)malloc(sz * 4 * sizeof(float));
	memset(addr, 0, sz * 4 * sizeof(float));
	return addr;
}

// 初始化小球的半径，位置等参数，并存在内存和gpu里
void Spheres::init_buffer() {
	cpu_pos = malset(cpu_pos, num_particles);
	cpu_vel = malset(cpu_vel, num_particles);
	create_gpu_buffer(&gpu_pos, num_particles * 4 * sizeof(float));
	create_gpu_buffer(&gpu_vel, num_particles * 4 * sizeof(float));
	create_gpu_buffer(&gpu_reod_pos, num_particles * 4 * sizeof(float));
	create_gpu_buffer(&gpu_reod_vel, num_particles * 4 * sizeof(float));
	create_gpu_buffer(&gpu_hash, num_particles * sizeof(uint));
	create_gpu_buffer(&gpu_index, num_particles * sizeof(uint));
	create_gpu_buffer(&gpu_cell_start, num_grid_cells * sizeof(uint));
	create_gpu_buffer(&gpu_cell_end, num_grid_cells * sizeof(uint));
	set_constants(&params);
}

// 初始化粒子系统
void Spheres::init_particle_params() {
	int p = 0, v = 0;
	for (uint i = 0; i < num_particles; i++) {
		cpu_pos[p++] = (float)rand() / (float)RAND_MAX;
		cpu_pos[p++] = (float)rand() / (float)RAND_MAX;
		cpu_pos[p++] = (float)rand() / (float)RAND_MAX;
		cpu_pos[p++] = (rand() % 20 + 10) / 1000.0;

		cpu_vel[v++] = (rand() % 30 + 10) / 1000.0;
		cpu_vel[v++] = (rand() % 30 + 10) / 1000.0;
		cpu_vel[v++] = (rand() % 30 + 10) / 1000.0;
		cpu_vel[v++] = 0;
	}
	set_gpu_data(POS, cpu_pos, 0, num_particles);
	set_gpu_data(VEL, cpu_vel, 0, num_particles);
}

// 刷新函数
void Spheres::update(float deltaTime) {
	refresh_particles(gpu_pos,gpu_vel,deltaTime,num_particles);
	reckon_hash(gpu_hash,gpu_index,gpu_pos,num_particles);
	merge_sort(gpu_hash, gpu_index, num_particles);
	find_cell_bounds_and_reorder(gpu_cell_start,gpu_cell_end,gpu_reod_pos,gpu_reod_vel,gpu_hash,gpu_index,gpu_pos,gpu_vel,num_particles,num_grid_cells);
	collide(gpu_vel,gpu_reod_pos,gpu_reod_vel,gpu_index,gpu_cell_start,gpu_cell_end,num_particles,num_grid_cells);
	cpu_pos = get_gpu_data(POS);
}

// 从gpu中拿到数据
float* Spheres::get_gpu_data(attr_type ty) {
	float* cpudata = 0;
	cl_mem gpudata = 0;
	if (ty == POS) {
		cpudata = cpu_pos;
		gpudata = gpu_pos;
	}
	else if(ty == VEL){
		cpudata = cpu_vel;
		gpudata = gpu_vel;
	}
	get_data_from_gpu(cpudata, gpudata, num_particles * 4 * sizeof(float));
	return cpudata;
}

// 将数据存在gpu里
void Spheres::set_gpu_data(attr_type ty, const float* data, int start, int count) {
	if (ty == POS) {
		set_data_on_gpu(gpu_pos, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
	}
	else if (ty == VEL) {
		set_data_on_gpu(gpu_vel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
	}
}

