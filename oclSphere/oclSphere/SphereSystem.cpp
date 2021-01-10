#include "SphereSystem.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <cl/cl.h>
#include <GL/glew.h>
#include <iostream>

typedef cl_mem cl_mem;
typedef unsigned int uint;

static cl_command_queue cqDefaultCommandQue;

static cl_mem params;

Spheres::Spheres(uint numParticles, uint3 gridSize) {
	num_particles = numParticles;
	cpu_pos = cpu_vel = 0;
	gpu_pos = gpu_vel = 0;
	grid_size = gridSize;
	init_params();
	init_data();
}

void Spheres::init_params() {
	num_grid_cells = grid_size.x * grid_size.y * grid_size.z;
	params.gridSize = grid_size;
	params.numCells = num_grid_cells;
	params.numBodies = num_particles;
	params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	params.particleRadius = 0.03f;
	params.cellSize = make_float3(params.particleRadius * 2.0f, params.particleRadius * 2.0f, params.particleRadius * 2.0f);
	params.spring = 0.2f;
	params.damping = 0.02f;
	params.shear = 0.01f;
	params.attraction = 0.0f;
	params.boundaryDamping = -0.5f;
	params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
	params.globalDamping = 1.0f;
}

void Spheres::init_data() {
	cpu_pos = (float*)malloc(num_particles * 4 * sizeof(float));
	cpu_vel = (float*)malloc(num_particles * 4 * sizeof(float));
	cpu_reco_pos = (float*)malloc(num_particles * 4 * sizeof(float));
	cpu_reco_vel = (float*)malloc(num_particles * 4 * sizeof(float));
	cpu_hash = (uint*)malloc(num_particles * sizeof(uint));
	cpu_index = (uint*)malloc(num_particles * sizeof(uint));
	cpu_cell_start = (uint*)malloc(num_grid_cells * sizeof(uint));
	cpu_cell_end = (uint*)malloc(num_grid_cells * sizeof(uint));

	memset(cpu_pos, 0, num_particles * 4 * sizeof(float));
	memset(cpu_vel, 0, num_particles * 4 * sizeof(float));
	memset(cpu_cell_start, 0, num_grid_cells * sizeof(uint));
	memset(cpu_cell_end, 0, num_grid_cells * sizeof(uint));

	create_gpu_buffer(&gpu_pos, num_particles * 4 * sizeof(float));
	create_gpu_buffer(&gpu_vel, num_particles * 4 * sizeof(float));
	create_gpu_buffer(&gpu_reco_pos, num_particles * 4 * sizeof(float));
	create_gpu_buffer(&gpu_reco_vel, num_particles * 4 * sizeof(float));
	create_gpu_buffer(&gpu_hash, num_particles * sizeof(uint));
	create_gpu_buffer(&gpu_index, num_particles * sizeof(uint));
	create_gpu_buffer(&gpu_cell_start, num_grid_cells * sizeof(uint));
	create_gpu_buffer(&gpu_cell_end, num_grid_cells * sizeof(uint));

	set_constants(&params);
}

void Spheres::reset() {
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
	set_array(POSITION, cpu_pos, 0, num_particles);
	set_array(VELOCITY, cpu_vel, 0, num_particles);
}

void Spheres::update(float deltaTime) {
	refresh_particles(gpu_pos,gpu_vel,deltaTime,num_particles);
	reckon_hash(gpu_hash,gpu_index,gpu_pos,num_particles);
	merge_sort(NULL, gpu_hash, gpu_index, gpu_hash, gpu_index, 1, num_particles, 0);
	find_cell_bounds_and_reorder(gpu_cell_start,gpu_cell_end,gpu_reco_pos,gpu_reco_vel,gpu_hash,gpu_index,gpu_pos,gpu_vel,num_particles,num_grid_cells);
	collide(gpu_vel,gpu_reco_pos,gpu_reco_vel,gpu_index,gpu_cell_start,gpu_cell_end,num_particles,num_grid_cells);
	cpu_pos = get_array(POSITION);
}

float* Spheres::get_array(ParticleArray array) {
	float* cpudata = 0;
	cl_mem gpudata = 0;
	switch (array) {
		case POSITION:
			cpudata = cpu_pos;
			gpudata = gpu_pos;
			break;
		case VELOCITY:
			cpudata = cpu_vel;
			gpudata = gpu_vel;
			break;
	}
	get_data_from_gpu(cpudata, gpudata, num_particles * 4 * sizeof(float));
	return cpudata;
}

void Spheres::set_array(ParticleArray array, const float* data, int start, int count) {
	switch (array) {
	case POSITION:
		set_data_on_gpu(gpu_pos, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	case VELOCITY:
		set_data_on_gpu(gpu_vel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	}
}

