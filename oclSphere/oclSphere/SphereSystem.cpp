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

Spheres::Spheres(uint numParticles, uint3 gridSize, float fParticleRadius, float fColliderRadius) :
	num_particles(numParticles),
	cpu_pos(0),
	cpu_vel(0),
	gpu_pos(0),
	gpu_vel(0),
	grid_size(gridSize)
{
	num_grid_cells = grid_size.x * grid_size.y * grid_size.z;

	params.gridSize = grid_size;
	params.numCells = num_grid_cells;
	params.numBodies = num_particles;
	params.particleRadius = fParticleRadius;
	params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	params.colliderRadius = fColliderRadius;

	params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	params.cellSize = make_float3(params.particleRadius * 2.0f, params.particleRadius * 2.0f, params.particleRadius * 2.0f);

	params.spring = 0.5f;
	params.damping = 0.02f;
	params.shear = 0.1f;
	params.attraction = 0.0f;
	params.boundaryDamping = -0.5f;

	params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
	params.globalDamping = 1.0f;

	initialize(numParticles);
}

void Spheres::initialize(int numParticles) {
	num_particles = numParticles;
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

	allocateArray(&gpu_pos, num_particles * 4 * sizeof(float));
	allocateArray(&gpu_vel, num_particles * 4 * sizeof(float));
	allocateArray(&gpu_reco_pos, num_particles * 4 * sizeof(float));
	allocateArray(&gpu_reco_vel, num_particles * 4 * sizeof(float));
	allocateArray(&gpu_hash, num_particles * sizeof(uint));
	allocateArray(&gpu_index, num_particles * sizeof(uint));
	allocateArray(&gpu_cell_start, num_grid_cells * sizeof(uint));
	allocateArray(&gpu_cell_end, num_grid_cells * sizeof(uint));

	setParameters(&params);
	setParametersHost(&params);
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
	integrateSystem(gpu_pos,gpu_vel,deltaTime,num_particles);

	calcHash(gpu_hash,gpu_index,gpu_pos,num_particles);

	bitonicSort(NULL, gpu_hash, gpu_index, gpu_hash, gpu_index, 1, num_particles, 0);

	findCellBoundsAndReorder(gpu_cell_start,gpu_cell_end,gpu_reco_pos,gpu_reco_vel,gpu_hash,gpu_index,gpu_pos,gpu_vel,num_particles,num_grid_cells);

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
	copyArrayFromDevice(cpudata, gpudata, num_particles * 4 * sizeof(float));
	return cpudata;
}

void Spheres::set_array(ParticleArray array, const float* data, int start, int count) {
	switch (array) {
	case POSITION:
		copyArrayToDevice(gpu_pos, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	case VELOCITY:
		copyArrayToDevice(gpu_vel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	}
}

