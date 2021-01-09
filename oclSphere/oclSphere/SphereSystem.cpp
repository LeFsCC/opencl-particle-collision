#include "SphereSystem.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <cl/cl.h>
#include <GL/glew.h>
#include <CL/cl.h>
#include <gl/GL.h>
#include <iostream>

typedef cl_mem memHandle_t;
typedef unsigned int uint;


//OpenCL particles kernels
//static cl_kernel ckIntegrate, ckCalcHash, ckMemset, ckFindCellBoundsAndReorder, ckCollide;

static cl_command_queue cqDefaultCommandQue;

static cl_mem params;

Spheres::Spheres(uint numParticles, uint3 gridSize, float fParticleRadius, float fColliderRadius) :
	m_bInitialized(false),
	m_numParticles(numParticles),
	m_hPos(0),
	m_hVel(0),
	m_dPos(0),
	m_dVel(0),
	m_gridSize(gridSize),
	m_solverIterations(1)
{
	m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
	float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

	m_gridSortBits = 18;    // increase this for larger grids

	// set simulation parameters
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.numBodies = m_numParticles;
	m_params.particleRadius = fParticleRadius;
	m_params.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
	m_params.colliderRadius = fColliderRadius;

	m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
	float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
	m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

	m_params.spring = 0.5f;
	m_params.damping = 0.02f;
	m_params.shear = 0.1f;
	m_params.attraction = 0.5f;
	m_params.boundaryDamping = -0.5f;

	m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
	m_params.globalDamping = 1.0f;

	_initialize(numParticles);
}

Spheres::~Spheres() {
	_finalize();
	m_numParticles = 0;
}

void Spheres::_finalize() {

	free(m_hPos);
	free(m_hVel);
	free(m_hCellStart);
	free(m_hCellEnd);
	free(m_hReorderedVel);
	free(m_hReorderedPos);
	free(m_hIndex);
	free(m_hHash);

	freeArray(m_dPos);
	freeArray(m_dVel);
	freeArray(m_dReorderedPos);
	freeArray(m_dReorderedVel);
	freeArray(m_dHash);
	freeArray(m_dIndex);
	freeArray(m_dCellStart);
	freeArray(m_dCellEnd);
}

inline float lerp(float a, float b, float t) {
	return a + t * (b - a);
}

void colorRamp(float t, float* r) {
	const int ncolors = 7;
	float c[ncolors][3] = {
		{ 1.0f, 0.0f, 0.0f, },
		{ 1.0f, 0.5f, 0.0f, },
		{ 1.0f, 1.0f, 0.0f, },
		{ 0.0f, 1.0f, 0.0f, },
		{ 0.0f, 1.0f, 1.0f, },
		{ 0.0f, 0.0f, 1.0f, },
		{ 1.0f, 0.0f, 1.0f, },
	};
	t = t * (ncolors - 1);
	int i = (int)t;
	float u = t - floor(t);
	r[0] = lerp(c[i][0], c[i + 1][0], u);
	r[1] = lerp(c[i][1], c[i + 1][1], u);
	r[2] = lerp(c[i][2], c[i + 1][2], u);
}


void Spheres::_initialize(int numParticles) {
	m_numParticles = numParticles;

	//Allocate host storage
	m_hPos = (float*)malloc(m_numParticles * 4 * sizeof(float));
	m_hVel = (float*)malloc(m_numParticles * 4 * sizeof(float));
	m_hReorderedPos = (float*)malloc(m_numParticles * 4 * sizeof(float));
	m_hReorderedVel = (float*)malloc(m_numParticles * 4 * sizeof(float));
	m_hHash = (uint*)malloc(m_numParticles * sizeof(uint));
	m_hIndex = (uint*)malloc(m_numParticles * sizeof(uint));
	m_hCellStart = (uint*)malloc(m_numGridCells * sizeof(uint));
	m_hCellEnd = (uint*)malloc(m_numGridCells * sizeof(uint));

	memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));
	memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));
	memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

	//Allocate GPU data
	allocateArray(&m_dPos, m_numParticles * 4 * sizeof(float));
	allocateArray(&m_dVel, m_numParticles * 4 * sizeof(float));
	allocateArray(&m_dReorderedPos, m_numParticles * 4 * sizeof(float));
	allocateArray(&m_dReorderedVel, m_numParticles * 4 * sizeof(float));
	allocateArray(&m_dHash, m_numParticles * sizeof(uint));
	allocateArray(&m_dIndex, m_numParticles * sizeof(uint));
	allocateArray(&m_dCellStart, m_numGridCells * sizeof(uint));
	allocateArray(&m_dCellEnd, m_numGridCells * sizeof(uint));

	setParameters(&m_params);
	setParametersHost(&m_params);
	m_bInitialized = true;
}

uint Spheres::createVBO(uint size) {
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

inline float frand(void) {
	return (float)rand() / (float)RAND_MAX;
}

void Spheres::reset(ParticleConfig config) {
	switch (config) {
		case CONFIG_RANDOM: {
			int p = 0, v = 0;
			for (uint i = 0; i < m_numParticles; i++)
			{
				float point[3];
				point[0] = frand();
				point[1] = frand();
				point[2] = frand();
				m_hPos[p++] = 2.0f * (point[0] - 0.5f);
				m_hPos[p++] = 2.0f * (point[1] - 0.5f);
				m_hPos[p++] = 2.0f * (point[2] - 0.5f);
				m_hPos[p++] = (rand() % 20 + 10) / 1000.0;

				m_hVel[v++] = (rand() % 30 + 10) / 1000.0;
				m_hVel[v++] = (rand() % 30 + 10) / 1000.0;
				m_hVel[v++] = (rand() % 30 + 10) / 1000.0;
				m_hVel[v++] = 0;
			}
			setArray(POSITION, m_hPos, 0, m_numParticles);
			setArray(VELOCITY, m_hVel, 0, m_numParticles);
		}
		break;
	}
}

//Step the simulation
void Spheres::update(float deltaTime) {

	setParameters(&m_params);
	setParametersHost(&m_params);

	integrateSystem(
		m_dPos,
		m_dVel,
		deltaTime,
		m_numParticles
	);

	calcHash(
		m_dHash,
		m_dIndex,
		m_dPos,
		m_numParticles
	);

	bitonicSort(NULL, m_dHash, m_dIndex, m_dHash, m_dIndex, 1, m_numParticles, 0);

	//Find start and end of each cell and
	//Reorder particle data for better cache coherency
	findCellBoundsAndReorder(
		m_dCellStart,
		m_dCellEnd,
		m_dReorderedPos,
		m_dReorderedVel,
		m_dHash,
		m_dIndex,
		m_dPos,
		m_dVel,
		m_numParticles,
		m_numGridCells
	);

	collide(
		m_dVel,
		m_dReorderedPos,
		m_dReorderedVel,
		m_dIndex,
		m_dCellStart,
		m_dCellEnd,
		m_numParticles,
		m_numGridCells
	);
	m_hPos = getArray(POSITION);
}

float* Spheres::getArray(ParticleArray array) {
	float* hdata = 0;
	memHandle_t ddata = 0;

	switch (array) {
	default:
	case POSITION:
		hdata = m_hPos;
		ddata = m_dPos;
		break;
	case VELOCITY:
		hdata = m_hVel;
		ddata = m_dVel;
		break;
	}
	copyArrayFromDevice(hdata, ddata, m_numParticles * 4 * sizeof(float));
	return hdata;
}

void Spheres::setArray(ParticleArray array, const float* data, int start, int count) {
	switch (array) {
	default:
	case POSITION:
		copyArrayToDevice(m_dPos, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	case VELOCITY:
		copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
		break;
	}
}

