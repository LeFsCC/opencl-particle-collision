#pragma once
#ifndef SPHERESYSTEM_H
#define SPHERESYSTEM_H

#include <cl/cl.h>

typedef cl_mem memHandle_t;
typedef unsigned int uint;
#define oclCheckErrorEX(a, b, c) __oclCheckErrorEX(a, b, c, __FILE__ , __LINE__) 
#define oclCheckError(a, b) oclCheckErrorEX(a, b, 0) 

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
} simParams_t;

static inline  float3 make_float3(float x, float y, float z) {
    float3 t; t.x = x; t.y = y; t.z = z; return t;
}

class Spheres {

public:
    enum ParticleConfig
    {
        CONFIG_RANDOM,
        CONFIG_GRID,
        _NUM_CONFIGS
    };

    enum ParticleArray
    {
        POSITION,
        VELOCITY,
    };
	Spheres(uint numParticles, uint3 gridSize, float fParticleRadius, float fColliderRadius);
	~Spheres();
    void _finalize();
    void _initialize(int numParticles);
    uint createVBO(uint size);

    void dumpGrid();
    void dumpParticles(uint start, uint count);

    void reset(ParticleConfig config);
    void update(float deltaTime);
    void setArray(ParticleArray array, const float* data, int start, int count);

    void setIterations(int i) { m_solverIterations = i; }
    void setDamping(float x) { m_params.globalDamping = x; }
    void setGravity(float x) { m_params.gravity = make_float3(0.0f, x, 0.0f); }
    void setCollideSpring(float x) { m_params.spring = x; }
    void setCollideDamping(float x) { m_params.damping = x; }
    void setCollideShear(float x) { m_params.shear = x; }
    void setCollideAttraction(float x) { m_params.attraction = x; }
    void setColliderPos(float3 x) { m_params.colliderPos = x; }

    float getParticleRadius() { return m_params.particleRadius; }
    float3 getColliderPos() { return m_params.colliderPos; }
    float getColliderRadius() { return m_params.colliderRadius; }
    uint3 getGridSize() { return m_params.gridSize; }
    float3 getWorldOrigin() { return m_params.worldOrigin; }
    float3 getCellSize() { return m_params.cellSize; }
    float* getPos() { return m_hPos; }

protected: // data
    bool m_bInitialized;
    uint m_numParticles;

    // CPU data
    float* m_hPos;
    float* m_hVel;
    float* m_hReorderedPos;
    float* m_hReorderedVel;
    uint* m_hCellStart;
    uint* m_hCellEnd;
    uint* m_hHash;
    uint* m_hIndex;

    // GPU data
    memHandle_t          m_dPos;
    memHandle_t          m_dVel;
    memHandle_t m_dReorderedPos;
    memHandle_t m_dReorderedVel;
    memHandle_t         m_dHash;
    memHandle_t        m_dIndex;
    memHandle_t    m_dCellStart;
    memHandle_t      m_dCellEnd;

    uint m_gridSortBits;
    uint       m_posVbo;
    uint     m_colorVBO;

    // params
    simParams_t m_params;
    uint3 m_gridSize;
    uint m_numGridCells;
    uint m_solverIterations;
};


extern "C" void integrateSystem(memHandle_t d_Pos, memHandle_t d_Vel, float deltaTime, uint numParticles);

extern "C" void calcHash( memHandle_t d_Hash, memHandle_t d_Index, memHandle_t d_Pos, int numParticles);

extern "C" void findCellBoundsAndReorder(memHandle_t d_CellStart, memHandle_t d_CellEnd, memHandle_t d_ReorderedPos,
	memHandle_t d_ReorderedVel, memHandle_t d_Hash, memHandle_t d_Index, memHandle_t d_Pos, memHandle_t d_Vel,
	uint numParticles, uint numCells);

extern "C" void collide(memHandle_t d_Vel, memHandle_t d_ReorderedPos, memHandle_t d_ReorderedVel, memHandle_t d_Index,
	memHandle_t d_CellStart, memHandle_t d_CellEnd, uint   numParticles, uint   numCells);

#endif SPHERESYSTEM_H